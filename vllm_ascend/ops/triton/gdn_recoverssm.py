# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP32 GDN checkpoints and accepted-prefix recovery on Ascend.

Q/K are normalized by the caller, matching the existing recurrent path.
Verification never overwrites the checkpoint. All records use the effective
post-normalization key and post-activation gates seen by verification.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, init_device_properties_triton

GDN_PAGE_COUNT: tl.constexpr = 5  # conv, checkpoint, correction, key, log-decay
RECOVERSSM_COMMIT_HEAD_TILE = 2
# Conservative UB budgets in BV x BK FP32 matrix equivalents. The GDN
# align path commits final and boundary states in one kernel, so it uses the
# larger boundary budget for the whole launch.
RECOVERSSM_COMMIT_UB_MATRICES = 2
RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES = 6
# Reserve tile rows for FP32 vectors and other temporary workspace.
RECOVERSSM_COMMIT_UB_HEADROOM_ROWS = 2
RECOVERSSM_VALUE_TILE = 64
RECOVERSSM_VALUE_TILE_LARGE = 128
RECOVERSSM_LARGE_TILE_MIN_PROGRAMS = 96
RECOVERSSM_CONV_TILE = 4096
RECOVERSSM_CONV_STRIDED_TILE = 256


def _inner_stride_signature(tensor):
    return tuple(stride if size > 1 else 0 for size, stride in zip(tensor.shape[1:], tensor.stride()[1:]))


def gdn_recoverssm_verify(
    q,
    k,
    v,
    g,
    beta,
    checkpoint_state,
    correction_cache,
    key_cache,
    gdn_gate_cache,
    query_start_loc,
    state_indices,
    spec_query_len,
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
    tensors = (
        q,
        k,
        v,
        g,
        beta,
        checkpoint_state,
        correction_cache,
        key_cache,
        gdn_gate_cache,
        query_start_loc,
        state_indices,
    )
    if any(t.device != q.device for t in tensors):
        raise ValueError("GDN RecoverSSM inputs must share a device")
    # The current rearrange/L2-norm path guarantees unit channel strides; guard other callers.
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("GDN RecoverSSM Q/K/V channels must be contiguous")
    rows = state_indices.numel()
    if state_indices.ndim != 1 or query_start_loc.shape != (rows + 1,):
        raise ValueError("GDN RecoverSSM request indices and offsets do not match")
    if tokens > rows * spec_query_len:
        raise ValueError("GDN RecoverSSM input exceeds its record capacity")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (state_indices, query_start_loc)):
        raise ValueError("GDN RecoverSSM request metadata must use integer tensors")
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    if rows and tokens:
        block_v = min(triton.next_power_of_2(dv), RECOVERSSM_VALUE_TILE)
        # Larger A5 tiles reduce repeated K-side loads when the remaining grid
        # still has enough programs to keep the device occupied.
        is_a5 = is_950()
        if (
            is_a5
            and block_v < RECOVERSSM_VALUE_TILE_LARGE
            and rows * hv >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS
        ):
            large_v = min(triton.next_power_of_2(dv), RECOVERSSM_VALUE_TILE_LARGE)
            if rows * hv * triton.cdiv(dv, large_v) >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
                block_v = large_v
        grid = (triton.cdiv(dv, block_v), hv, rows)
        _recoverssm_verify_kernel[grid](
            # Inputs, persistent checkpoint, and per-token records.
            q_ptr=q,
            k_ptr=k,
            v_ptr=v,
            gate_ptr=g,
            beta_ptr=beta,
            A_log_ptr=None,
            dt_bias_ptr=None,
            state_ptr=checkpoint_state,
            correction_cache_ptr=correction_cache,
            # The shared KDA/GDN kernel uses this slot for KDA's packed
            # key/gate records; GDN stores only normalized keys here.
            kg_cache_ptr=key_cache,
            gdn_gate_cache_ptr=gdn_gate_cache,
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
            state_strides=checkpoint_state.stride(),
            correction_strides=correction_cache.stride(),
            kg_cache_strides=key_cache.stride(),
            gdn_gate_cache_strides=gdn_gate_cache.stride(),
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
            BV=block_v,
            INPUT_MODE=GDN_MODE,
            USE_LOWER_BOUND=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@dataclass(frozen=True)
class GDNRecoverSSMLayerCache:
    """Cache tensors and kernel address metadata for one GDN layer."""

    conv: torch.Tensor
    conv_base_addr: int
    conv_block_stride: int
    conv_dim_stride: int
    conv_token_stride: int
    checkpoint: torch.Tensor
    checkpoint_base_addr: int
    checkpoint_block_stride: int
    correction: torch.Tensor
    correction_base_addr: int
    correction_block_stride: int
    key: torch.Tensor
    key_base_addr: int
    key_block_stride: int
    log_decay: torch.Tensor
    log_decay_base_addr: int
    log_decay_block_stride: int

    @classmethod
    def from_layer(cls, layer: Any) -> "GDNRecoverSSMLayerCache":
        if len(layer.kv_cache) != GDN_PAGE_COUNT:
            raise ValueError("GDN RecoverSSM requires conv, checkpoint, correction, key and gate pages")
        conv, checkpoint, correction, key, log_decay = layer.kv_cache
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        return cls(
            conv=conv,
            conv_base_addr=conv.data_ptr(),
            conv_block_stride=conv.stride(0),
            conv_dim_stride=conv.stride(1),
            conv_token_stride=conv.stride(2),
            checkpoint=checkpoint,
            correction=correction,
            key=key,
            log_decay=log_decay,
            checkpoint_base_addr=checkpoint.data_ptr(),
            checkpoint_block_stride=checkpoint.stride(0),
            correction_base_addr=correction.data_ptr(),
            correction_block_stride=correction.stride(0),
            key_base_addr=key.data_ptr(),
            key_block_stride=key.stride(0),
            log_decay_base_addr=log_decay.data_ptr(),
            log_decay_block_stride=log_decay.stride(0),
        )

    def state_pages(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.checkpoint, self.correction, self.key, self.log_decay

    def all_pages(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.conv, *self.state_pages()


def _create_state_commit_cache_address_stride_table(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> torch.Tensor:
    """Create the cache-kind-major address/stride table consumed by the kernel."""
    return torch.tensor(
        [
            [
                [cache.checkpoint_base_addr, cache.checkpoint_block_stride],
                [cache.correction_base_addr, cache.correction_block_stride],
                [cache.key_base_addr, cache.key_block_stride],
                [cache.log_decay_base_addr, cache.log_decay_block_stride],
            ]
            for cache in layer_caches
        ],
        dtype=torch.int64,
        device=layer_caches[0].checkpoint.device,
    ).permute(1, 2, 0).contiguous()


def _create_conv_cache_address_stride_table(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> torch.Tensor:
    """Create the field-major convolution address/stride table for the kernel."""
    return torch.tensor(
        [
            [
                cache.conv_base_addr,
                cache.conv_block_stride,
                cache.conv_dim_stride,
                cache.conv_token_stride,
            ]
            for cache in layer_caches
        ],
        dtype=torch.int64,
        device=layer_caches[0].conv.device,
    ).T.contiguous()


def _get_shared_conv_dim_stride(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> int | None:
    """Return the common dim stride, or None when layers use different strides."""
    reference_stride = layer_caches[0].conv_dim_stride
    if all(cache.conv_dim_stride == reference_stride for cache in layer_caches):
        return reference_stride
    return None


@dataclass(frozen=True)
class GDNRecoverSSMCommitPlan:
    """Reusable per-request buffers populated by the commit-plan kernel."""

    commit_lens: torch.Tensor
    state_commit_lens: torch.Tensor
    boundary_commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor

    @classmethod
    def allocate(cls, max_num_reqs: int, device: torch.device) -> "GDNRecoverSSMCommitPlan":
        # Allocate one reusable contiguous buffer for six per-request plan
        # arrays: commit length, non-boundary/boundary commit lengths, final and
        # boundary state indices, and boundary recovery length. The row slices
        # below are views sharing this allocation. The plan kernel overwrites
        # every row in [:batch] before use, so uninitialized capacity beyond the
        # current batch is safe and avoids an unnecessary device-side fill.
        #
        #                                        request 0   request 1   request 2   request 3
        #                                      ┌───────────┬───────────┬───────────┬───────────┐
        # storage[0] commit_lens               │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[1] state_commit_lens         │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[2] boundary_commit_lens      │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[3] final_state_indices       │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[4] boundary_state_indices    │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[5] boundary_recovery_lens    │           │           │           │           │
        #                                      └───────────┴───────────┴───────────┴───────────┘
        storage = torch.empty((6, max_num_reqs), dtype=torch.int32, device=device)
        return cls(
            commit_lens=storage[0],
            state_commit_lens=storage[1],
            boundary_commit_lens=storage[2],
            final_state_indices=storage[3],
            boundary_state_indices=storage[4],
            boundary_recovery_lens=storage[5],
        )

    @property
    def capacity(self) -> int:
        return self.commit_lens.shape[0]


def _validate_layer_caches(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
    spec_query_len: int,
) -> int:
    reference = layer_caches[0]
    blocks, num_value_heads, value_dim, key_dim = reference.checkpoint.shape
    num_key_heads = reference.key.shape[1]
    if num_key_heads < 1 or num_value_heads % num_key_heads:
        raise ValueError("GDN RecoverSSM requires integral GQA")

    channels = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    if (
        reference.conv.ndim != 3
        or reference.conv.shape[:2] != (blocks, channels)
        or reference.conv.device != reference.checkpoint.device
    ):
        raise ValueError("GDN RecoverSSM convolution page has incompatible layout")
    history_len = reference.conv.shape[2] - spec_query_len + 1
    if history_len < 1:
        raise ValueError("GDN RecoverSSM convolution history is too short")

    for current in layer_caches[1:]:
        if any(
            (tensor.shape, tensor.dtype, tensor.device)
            != (reference_tensor.shape, reference_tensor.dtype, reference_tensor.device)
            for tensor, reference_tensor in zip(current.all_pages(), reference.all_pages())
        ):
            raise ValueError("GDN RecoverSSM layers must share shapes, dtypes and device")
        for name, tensor, reference_tensor in zip(
            ("checkpoint", "correction", "key", "gate"),
            current.state_pages(),
            reference.state_pages(),
        ):
            if _inner_stride_signature(tensor) != _inner_stride_signature(reference_tensor):
                raise ValueError(f"GDN RecoverSSM layers must share {name} page inner strides")
    return history_len


@dataclass
class GDNRecoverSSMCommitContext:
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...]
    commit_plan: GDNRecoverSSMCommitPlan
    spec_query_len: int
    # Number of convolution-history tokens retained after committing the
    # accepted prefix: conv_window_len - spec_query_len + 1.
    conv_history_len: int
    # Common conv dim stride used as a compile-time kernel constant. None
    # makes the kernel load each layer's stride from
    # _conv_cache_address_stride_table.
    conv_shared_dim_stride: int | None
    # Contiguous device-side int64 table with shape [4, num_layers]. Column i
    # describes layer_caches[i].conv:
    #   [0, i] = conv.data_ptr(), the base address of layer i's conv cache.
    #   [1, i] = conv.stride(0), the cache-block stride in elements.
    #   [2, i] = conv.stride(1), the channel/dim stride in elements.
    #   [3, i] = conv.stride(2), the token/history stride in elements.
    # The compact-conv kernel indexes each row with its device-side layer id.
    _conv_cache_address_stride_table: torch.Tensor = field(repr=False)
    # Contiguous device-side int64 table with shape [4, 2, num_layers]. For
    # layer_caches[i], every block stride below is measured in elements:
    #   [0, 0, i] = checkpoint.data_ptr(), the checkpoint-cache base address.
    #   [0, 1, i] = checkpoint.stride(0), its cache-block stride.
    #   [1, 0, i] = correction.data_ptr(), the correction-cache base address.
    #   [1, 1, i] = correction.stride(0), its cache-block stride.
    #   [2, 0, i] = key.data_ptr(), the key-cache base address.
    #   [2, 1, i] = key.stride(0), its cache-block stride.
    #   [3, 0, i] = log_decay.data_ptr(), the log-decay-cache base address.
    #   [3, 1, i] = log_decay.stride(0), its cache-block stride.
    # The state-commit kernel uses these addresses and strides to reach every
    # layer while processing all layers in one launch.
    _state_commit_cache_address_stride_table: torch.Tensor = field(repr=False)

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
    ) -> "GDNRecoverSSMCommitContext":
        """Bind named layer caches and allocate reusable commit metadata."""
        if not layers or spec_query_len < 1 or max_num_reqs < 1:
            raise ValueError("GDN RecoverSSM requires layers and positive capacities")
        layer_caches = tuple(GDNRecoverSSMLayerCache.from_layer(layer) for layer in layers)
        history_len = _validate_layer_caches(layer_caches, spec_query_len)
        device = layer_caches[0].checkpoint.device
        if device.type == "npu":
            init_device_properties_triton()
        return cls(
            layer_caches=layer_caches,
            commit_plan=GDNRecoverSSMCommitPlan.allocate(max_num_reqs, device),
            spec_query_len=spec_query_len,
            conv_history_len=history_len,
            conv_shared_dim_stride=_get_shared_conv_dim_stride(layer_caches),
            _conv_cache_address_stride_table=_create_conv_cache_address_stride_table(layer_caches),
            _state_commit_cache_address_stride_table=_create_state_commit_cache_address_stride_table(layer_caches),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Commit each request's accepted prefix into every bound GDN layer."""
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
        if batch > self.commit_plan.capacity or state_indices.ndim != 1 or query_start_loc.shape != (batch + 1,):
            raise ValueError("GDN RecoverSSM commit request metadata exceeds capacity or has incompatible shape")
        if request_indices is not None and request_indices.shape != (batch,):
            raise ValueError("GDN RecoverSSM sampler mapping must have one row per request")
        if request_indices is None and num_accepted_tokens.numel() < batch:
            raise ValueError("GDN RecoverSSM acceptance counts are too short")
        reference = self.layer_caches[0]
        device = reference.checkpoint.device
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

        conv = reference.conv
        state = reference.checkpoint
        keys = reference.key
        plan = self.commit_plan
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
            plan.commit_lens,
            plan.state_commit_lens,
            plan.boundary_commit_lens,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.boundary_recovery_lens,
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
        # Strided channel gathers need smaller tiles to fit the A3 UB.
        conv_block = (
            RECOVERSSM_CONV_TILE
            if is_950() or self.conv_shared_dim_stride == 1
            else RECOVERSSM_CONV_STRIDED_TILE
        )
        conv_cache_address_stride_table = self._conv_cache_address_stride_table
        num_layers = len(self.layer_caches)
        _compact_conv_state_kernel[(triton.cdiv(conv.shape[1], conv_block), batch, num_layers)](
            conv,
            conv_cache_address_stride_table[0],
            conv_cache_address_stride_table[1],
            conv_cache_address_stride_table[2],
            conv_cache_address_stride_table[3],
            state_indices,
            plan.commit_lens,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.boundary_recovery_lens,
            0,
            conv.shape[1],
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=conv_block,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
            ALIGN_MODE=align_mode,
            DIM_STRIDE=self.conv_shared_dim_stride,
            USE_1D_CHANNEL_GATHER=not is_950(),
            num_warps=4,
        )
        # 3. Compose accepted records backwards, then merge the checkpoint.
        corrections = reference.correction
        gates = reference.log_decay
        state_commit_cache_address_stride_table = self._state_commit_cache_address_stride_table
        block_k = triton.next_power_of_2(dk)
        matrix_element_size = corrections.element_size()
        is_a5 = is_950()
        grouped = is_a5 and dk == dv == 128 and hv >= 2 and hv % RECOVERSSM_COMMIT_HEAD_TILE == 0
        value_tile = RECOVERSSM_VALUE_TILE
        if grouped:
            head_groups = hv // RECOVERSSM_COMMIT_HEAD_TILE
            commit_kernel = _commit_recoverssm_state_grouped_kernel
            options = {}
        else:
            # Match KDA's larger A5 ordinary-commit tile when the launch does
            # not also carry boundary accumulators.
            if is_a5 and not align_mode:
                value_tile *= 2
            head_groups = hv
            commit_kernel = _commit_recoverssm_state_kernel
            options = {"UNROLL_COMMIT": is_a5}

        # KDA launches ordinary and boundary commits separately. GDN keeps one
        # launch, so every aligned program is compiled with the final/boundary
        # accumulators and must use the boundary UB budget. The grouped kernel
        # holds two heads' matrices in each program.
        ub_matrices = RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES if align_mode else RECOVERSSM_COMMIT_UB_MATRICES
        bytes_per_row = ub_matrices * block_k * matrix_element_size
        if grouped:
            bytes_per_row *= RECOVERSSM_COMMIT_HEAD_TILE
        max_rows = max(1, get_ub_size_bytes() // bytes_per_row - RECOVERSSM_COMMIT_UB_HEADROOM_ROWS)
        value_tile = min(value_tile, 1 << (max_rows.bit_length() - 1))
        block_v = min(triton.next_power_of_2(dv), value_tile)
        # Explicit flat indexing keeps request rows outermost on both 3.2/3.6
        # backends, balancing mixed boundary requests without a new static flag.
        grid = (triton.cdiv(dv, block_v) * num_layers * head_groups * batch,)
        commit_kernel[grid](
            state_ref_ptr=state,
            state_base_addrs_ptr=state_commit_cache_address_stride_table[0, 0],
            state_block_strides_ptr=state_commit_cache_address_stride_table[0, 1],
            correction_cache_ref_ptr=corrections,
            correction_cache_base_addrs_ptr=state_commit_cache_address_stride_table[1, 0],
            correction_cache_block_strides_ptr=state_commit_cache_address_stride_table[1, 1],
            kg_cache_ref_ptr=keys,
            kg_cache_base_addrs_ptr=state_commit_cache_address_stride_table[2, 0],
            kg_cache_block_strides_ptr=state_commit_cache_address_stride_table[2, 1],
            A_log_ptr=None,
            dt_bias_ptr=None,
            state_indices_ptr=state_indices,
            commit_lens_ptr=plan.commit_lens,
            final_state_indices_ptr=plan.final_state_indices,
            boundary_state_indices_ptr=plan.boundary_state_indices,
            boundary_recovery_lens_ptr=plan.boundary_recovery_lens,
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
            num_layers=num_layers,
            K=dk,
            V=dv,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=hv,
            USE_LOWER_BOUND=False,
            ALIGN_MODE=align_mode,
            SPEC_QUERY_LEN=self.spec_query_len,
            gate_cache_base_addrs_ptr=state_commit_cache_address_stride_table[3, 0],
            gate_cache_block_strides_ptr=state_commit_cache_address_stride_table[3, 1],
            stride_gate_cache_head=gates.stride(1),
            stride_gate_cache_pos=gates.stride(2),
            NUM_KEY_HEADS=keys.shape[1],
            INPUT_MODE=GDN_MODE,
            num_warps=4,
            enable_fp_fusion=False,
            **options,
        )
