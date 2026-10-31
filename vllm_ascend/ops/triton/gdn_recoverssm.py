# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP32 GDN checkpoints and accepted-prefix recovery on Ascend.

Q/K are normalized by the caller, matching the existing recurrent path.
Verification never overwrites the checkpoint. All records use the effective
post-normalization key and post-activation gates seen by verification.
"""

from dataclasses import dataclass

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.recoverssm_common import (
    GDN_MODE,
    _compact_conv_state_kernel,
    _prepare_commit_plan_kernel,
    _commit_recoverssm_state_grouped_kernel,
    _commit_recoverssm_state_kernel,
    _recoverssm_verify_kernel,
)

PAGE_DESCRIPTOR_FIELDS: tl.constexpr = 5  # address followed by up to four strides
GDN_PAGE_COUNT: tl.constexpr = 5  # conv, checkpoint, correction, key, log-decay


def _check_dense_page(tensor, shape, device):
    if tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != device:
        raise ValueError(f"GDN RecoverSSM requires FP32 page {shape} on {device}")
    stride = 1
    for dim in range(tensor.ndim - 1, 0, -1):
        if tensor.stride(dim) != stride:
            raise ValueError("GDN RecoverSSM page inner dimensions must be contiguous")
        stride *= tensor.shape[dim]
    if tensor.stride(0) < stride:
        raise ValueError("GDN RecoverSSM pages must not overlap")


def gdn_recoverssm_verify(
    q, k, v, g, beta, checkpoint, corrections, keys, gates, query_start_loc, state_indices, spec_query_len
):
    """Return GDN outputs and compact records, leaving the checkpoint intact."""
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("GDN RecoverSSM Q/K must have matching [1,T,HK,K] shapes")
    _, tokens, hk, dk = q.shape
    if v.ndim != 4 or v.shape[:2] != (1, tokens):
        raise ValueError("GDN RecoverSSM V must have shape [1,T,HV,V]")
    hv, dv = v.shape[2:]
    if hk < 1 or hv % hk or spec_query_len < 1:
        raise ValueError("GDN RecoverSSM requires integral GQA and a positive window")
    if g.shape != (1, tokens, hv) or beta.shape != g.shape:
        raise ValueError("GDN RecoverSSM requires one gate and beta per value head")
    tensors = (q, k, v, g, beta, checkpoint, corrections, keys, gates, query_start_loc, state_indices)
    if any(t.device != q.device for t in tensors):
        raise ValueError("GDN RecoverSSM inputs must share a device")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("GDN RecoverSSM Q/K/V channels must be contiguous")
    blocks = checkpoint.shape[0]
    _check_dense_page(checkpoint, (blocks, hv, dv, dk), q.device)
    _check_dense_page(corrections, (blocks, hv, spec_query_len, dv), q.device)
    _check_dense_page(keys, (blocks, hk, spec_query_len, dk), q.device)
    _check_dense_page(gates, (blocks, hv, spec_query_len), q.device)
    rows = state_indices.numel()
    if state_indices.ndim != 1 or query_start_loc.shape != (rows + 1,):
        raise ValueError("GDN RecoverSSM request indices and offsets do not match")
    if tokens > rows * spec_query_len:
        raise ValueError("GDN RecoverSSM input exceeds its record capacity")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (state_indices, query_start_loc)):
        raise ValueError("GDN RecoverSSM request metadata must use integer tensors")
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    if rows and tokens:
        _recoverssm_verify_kernel[(triton.cdiv(dv, 16), hv, rows)](
            # Inputs, persistent checkpoint, and per-token records.
            q_ptr=q,
            k_ptr=k,
            v_ptr=v,
            gate_ptr=g,
            beta_ptr=beta,
            A_log_ptr=None,
            dt_bias_ptr=None,
            state_ptr=checkpoint,
            correction_cache_ptr=corrections,
            kg_cache_ptr=keys,
            gate_cache_ptr=gates,
            query_start_loc_ptr=query_start_loc,
            state_indices_ptr=state_indices,
            out_ptr=out,
            lower_bound=0.0,
            null_block_id=0,
            # Activation and checkpoint layouts.
            stride_q_token=q.stride(1),
            stride_q_head=q.stride(2),
            stride_k_token=k.stride(1),
            stride_k_head=k.stride(2),
            stride_v_token=v.stride(1),
            stride_v_head=v.stride(2),
            stride_gate_token=g.stride(1),
            stride_gate_head=g.stride(2),
            stride_beta_token=beta.stride(1),
            stride_beta_head=beta.stride(2),
            state_strides=checkpoint.stride(),
            correction_strides=corrections.stride(),
            kg_cache_strides=keys.stride(),
            gate_cache_strides=gates.stride(),
            stride_out_token=out.stride(1),
            stride_out_head=out.stride(2),
            stride_query_start_loc=query_start_loc.stride(0),
            stride_state_indices=state_indices.stride(0),
            # Compile-time shape and input-format specialization.
            NUM_KEY_HEADS=hk,
            NUM_VALUE_HEADS=hv,
            K=dk,
            V=dv,
            BK=triton.next_power_of_2(dk),
            BV=16,
            INPUT_MODE=GDN_MODE,
            USE_LOWER_BOUND=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@dataclass
class GDNRecoverSSMCommitContext:
    pages: tuple
    descriptors: torch.Tensor
    state_layout: torch.Tensor
    spec_query_len: int
    max_num_reqs: int
    conv_history_len: int
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_dim_stride: int | None
    commit_lens: torch.Tensor
    state_commit_lens: torch.Tensor
    boundary_commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor

    @classmethod
    def create(cls, layers, *, spec_query_len, max_num_reqs):
        if not layers or spec_query_len < 1 or max_num_reqs < 1:
            raise ValueError("GDN RecoverSSM requires layers and positive capacities")
        pages = []
        descriptors = []
        for layer in layers:
            if len(layer.kv_cache) != GDN_PAGE_COUNT:
                raise ValueError("GDN RecoverSSM requires conv, checkpoint, correction, key and gate pages")
            conv, state, correction, keys, gates = layer.kv_cache
            conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
            blocks, hv, dv, dk = state.shape
            hk = keys.shape[1]
            if hk < 1 or hv % hk:
                raise ValueError("GDN RecoverSSM requires integral GQA")
            _check_dense_page(state, (blocks, hv, dv, dk), state.device)
            _check_dense_page(correction, (blocks, hv, spec_query_len, dv), state.device)
            _check_dense_page(keys, (blocks, hk, spec_query_len, dk), state.device)
            _check_dense_page(gates, (blocks, hv, spec_query_len), state.device)
            channels = 2 * hk * dk + hv * dv
            if conv.ndim != 3 or conv.shape[:2] != (blocks, channels) or conv.device != state.device:
                raise ValueError("GDN RecoverSSM convolution page has incompatible layout")
            history = conv.shape[2] - spec_query_len + 1
            if history < 1:
                raise ValueError("GDN RecoverSSM convolution history is too short")
            page = (conv, state, correction, keys, gates)
            if pages and any(
                (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device) for a, b in zip(page, pages[0])
            ):
                raise ValueError("GDN RecoverSSM layers must share shapes, dtypes and device")
            pages.append(page)
            for tensor in page:
                descriptors.append(
                    [tensor.data_ptr(), *tensor.stride(), *([0] * (PAGE_DESCRIPTOR_FIELDS - 1 - tensor.ndim))]
                )
        device = pages[0][1].device
        conv_states = [page[0] for page in pages]
        # Shared convolution kernel consumes contiguous per-layer arrays.
        conv_layout = torch.tensor(
            [[conv.data_ptr(), *conv.stride()] for conv in conv_states], dtype=torch.int64, device=device
        ).T.contiguous()
        # [cache kind, address/block stride, layer], with contiguous layer arrays.
        state_layout = torch.tensor(
            [[[page[kind].data_ptr(), page[kind].stride(0)] for page in pages] for kind in range(1, 5)],
            dtype=torch.int64,
            device=device,
        ).transpose(1, 2).contiguous()
        plan = torch.empty((6, max_num_reqs), dtype=torch.int32, device=device)
        return cls(
            pages=tuple(pages),
            descriptors=torch.tensor(descriptors, dtype=torch.int64, device=device),
            state_layout=state_layout,
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            conv_history_len=history,
            conv_state_base_addrs=conv_layout[0],
            conv_state_block_strides=conv_layout[1],
            conv_state_dim_strides=conv_layout[2],
            conv_state_token_strides=conv_layout[3],
            conv_dim_stride=(
                conv_states[0].stride(1)
                if all(conv.stride(1) == conv_states[0].stride(1) for conv in conv_states)
                else None
            ),
            commit_lens=plan[0],
            state_commit_lens=plan[1],
            boundary_commit_lens=plan[2],
            final_state_indices=plan[3],
            boundary_state_indices=plan[4],
            boundary_recovery_lens=plan[5],
        )

    def commit(
        self,
        num_accepted_tokens,
        state_indices,
        query_start_loc,
        *,
        request_indices=None,
        block_table=None,
        num_computed_tokens=None,
        mamba_block_size=None,
    ):
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(arg is None for arg in align_args):
            raise ValueError("GDN RecoverSSM align metadata is incomplete")
        align_mode = block_table is not None
        if align_mode:
            if mamba_block_size < self.spec_query_len:
                raise ValueError("GDN RecoverSSM align block size must cover one speculative window")
            if block_table.ndim != 2 or block_table.shape[1] < 1:
                raise ValueError("GDN RecoverSSM block table must be nonempty and two-dimensional")
            if block_table.dtype not in (torch.int32, torch.int64):
                raise ValueError("GDN RecoverSSM block table must use integer indices")
        batch = state_indices.numel()
        if batch > self.max_num_reqs or state_indices.ndim != 1 or query_start_loc.shape != (batch + 1,):
            raise ValueError("GDN RecoverSSM commit request metadata exceeds capacity or has incompatible shape")
        if request_indices is not None and request_indices.shape != (batch,):
            raise ValueError("GDN RecoverSSM sampler mapping must have one row per request")
        integer_tensors = (num_accepted_tokens, state_indices, query_start_loc)
        if request_indices is not None:
            integer_tensors += (request_indices,)
        if align_mode:
            integer_tensors += (num_computed_tokens,)
        if any(t.ndim != 1 or t.dtype not in (torch.int32, torch.int64) for t in integer_tensors):
            raise ValueError("GDN RecoverSSM commit metadata must be one-dimensional integer tensors")
        if request_indices is None and num_accepted_tokens.numel() < batch:
            raise ValueError("GDN RecoverSSM acceptance counts are too short")
        device = self.descriptors.device
        if any(t.device != device for t in (num_accepted_tokens, state_indices, query_start_loc)):
            raise ValueError("GDN RecoverSSM commit tensors must share the cache device")
        if request_indices is not None and request_indices.device != device:
            raise ValueError("GDN RecoverSSM sampler mapping must be on the cache device")
        if align_mode:
            if block_table.device != device or num_computed_tokens.device != device:
                raise ValueError("GDN RecoverSSM align metadata must be on the cache device")
            if request_indices is None and (block_table.shape[0] < batch or num_computed_tokens.numel() < batch):
                raise ValueError("GDN RecoverSSM align metadata must cover the sampler rows")
        if not batch:
            return
        conv, state, _, keys, _ = self.pages[0]
        _, hv, dv, dk = state.shape
        # 1. Resolve accepted lengths and both destinations once per request.
        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.state_commit_lens,
            self.boundary_commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            0,
            mamba_block_size or 1,
            block_table.shape[1] if align_mode else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_tokens.stride(0) if align_mode else 0,
            SPEC_QUERY_LEN=self.spec_query_len,
            INPUT_MODE=GDN_MODE,
            num_warps=1,
        )
        # 2. Copy final and optional boundary convolution histories.
        conv_block = 128
        _compact_conv_state_kernel[(triton.cdiv(conv.shape[1], conv_block), batch, len(self.pages))](
            conv,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            0,
            conv.shape[1],
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=conv_block,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align_mode,
            DIM_STRIDE=self.conv_dim_stride,
            USE_1D_CHANNEL_GATHER=not is_950(),
            num_warps=4,
        )
        # 3. Compose accepted records backwards, then merge the checkpoint.
        _, _, corrections, keys, gates = self.pages[0]
        grouped = is_950() and dk == dv == 128 and hv >= 2 and hv % 2 == 0
        kernel = _commit_recoverssm_state_grouped_kernel if grouped else _commit_recoverssm_state_kernel
        # Keep the small GDN value tile for both paths until device tuning.
        block_v = 16
        head_groups = hv // 2 if grouped else hv
        options = {} if grouped else {"UNROLL_COMMIT": is_950()}
        kernel[(triton.cdiv(dv, block_v) * len(self.pages) * head_groups * batch,)](
            state_ref_ptr=state,
            state_base_addrs_ptr=self.state_layout[0, 0],
            state_block_strides_ptr=self.state_layout[0, 1],
            correction_cache_ref_ptr=corrections,
            correction_cache_base_addrs_ptr=self.state_layout[1, 0],
            correction_cache_block_strides_ptr=self.state_layout[1, 1],
            kg_cache_ref_ptr=keys,
            kg_cache_base_addrs_ptr=self.state_layout[2, 0],
            kg_cache_block_strides_ptr=self.state_layout[2, 1],
            A_log_ptr=None,
            dt_bias_ptr=None,
            state_indices_ptr=state_indices,
            commit_lens_ptr=self.commit_lens,
            final_state_indices_ptr=self.final_state_indices,
            boundary_state_indices_ptr=self.boundary_state_indices,
            boundary_recovery_lens_ptr=self.boundary_recovery_lens,
            lower_bound=0.0,
            null_block_id=0,
            stride_state_head=state.stride(1),
            stride_state_v=state.stride(2),
            stride_state_k=state.stride(3),
            stride_correction_cache_head=corrections.stride(1),
            stride_correction_cache_pos=corrections.stride(2),
            stride_correction_cache_dim=corrections.stride(3),
            stride_kg_cache_head=keys.stride(1),
            stride_kg_cache_pos=keys.stride(2),
            stride_kg_cache_dim=keys.stride(3),
            stride_A_layer=0,
            stride_A_head=0,
            stride_dt_bias_layer=0,
            stride_dt_bias_head=0,
            stride_dt_bias_dim=0,
            stride_state_indices=state_indices.stride(0),
            num_layers=len(self.pages),
            K=dk,
            V=dv,
            BK=triton.next_power_of_2(dk),
            BV=block_v,
            NUM_HEADS=hv,
            USE_LOWER_BOUND=False,
            ALIGN_MODE=align_mode,
            SPEC_QUERY_LEN=self.spec_query_len,
            gate_cache_base_addrs_ptr=self.state_layout[3, 0],
            gate_cache_block_strides_ptr=self.state_layout[3, 1],
            stride_gate_cache_head=gates.stride(1),
            stride_gate_cache_pos=gates.stride(2),
            NUM_KEY_HEADS=keys.shape[1],
            INPUT_MODE=GDN_MODE,
            num_warps=4,
            enable_fp_fusion=False,
            **options,
        )
