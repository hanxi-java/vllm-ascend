# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend RecoverSSM verify and batched accepted-state recovery.

Adapted from vLLM c4df37dfcc. Preserve its paged-cache strides and public
verify/commit contracts. Q/K normalization follows the AscendC KDA kernel.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.recoverssm_common import (
    KDA_MODE,
    _compact_conv_state_kernel,
    _kda_gate,
    _l2_normalize_vector,
    _prepare_commit_plan_kernel,
    _commit_recoverssm_state_grouped_kernel,
    _commit_recoverssm_state_kernel,
    _recoverssm_verify_kernel,
)
from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes, init_device_properties_triton

RECOVERSSM_COMMIT_HEAD_TILE = 2
# Conservative UB budgets in BV x BK FP32 matrix equivalents. Boundary recovery
# includes final/boundary accumulators and merge temporaries; these estimates
# are not compiler-reported live-buffer counts.
RECOVERSSM_COMMIT_UB_MATRICES = 2
RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES = 6
# Reserve tile rows for FP32 vectors and other temporary workspace.
RECOVERSSM_COMMIT_UB_HEADROOM_ROWS = 2
RECOVERSSM_VALUE_TILE = 64
RECOVERSSM_VALUE_TILE_LARGE = 128
RECOVERSSM_LARGE_TILE_MIN_PROGRAMS = 96
RECOVERSSM_CONV_TILE = 4096
RECOVERSSM_CONV_STRIDED_TILE = 256


@triton.jit
def _kda_recurrent_step(
    state,
    k,
    v,
    raw_g,
    raw_beta,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    normalized_k = _l2_normalize_vector(k)
    gate = _kda_gate(
        raw_g,
        dt_bias,
        A,
        lower_bound,
        USE_LOWER_BOUND,
    )

    tl.static_assert(gate.dtype == tl.float32)
    state *= tl.exp(gate)[None, :]
    correction = v - tl.sum(state * normalized_k[None, :], axis=1)
    tl.static_assert(raw_beta.dtype == tl.float32)
    correction *= tl.sigmoid(raw_beta)
    return state + correction[:, None] * normalized_k[None, :], correction


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    num_blocks = checkpoint_state.shape[0]
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    expected_correction_shape = (
        num_blocks,
        num_heads,
        spec_query_len,
        value_dim,
    )
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(f"KDA RecoverSSM correction buffer needs shape {expected_correction_shape}")
    expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
    if kg_cache.shape != expected_kg_shape:
        raise ValueError(f"KDA RecoverSSM key/gate buffer needs shape {expected_kg_shape}")
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM correction buffer must use float32")
    if kg_cache.dtype != k.dtype:
        raise ValueError("KDA RecoverSSM key/gate buffer must match activation dtype")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError("KDA RecoverSSM speculative decode input exceeds its activation capacity")
    if out is None:
        out = torch.empty_like(v)
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    device = q.device
    if any(
        tensor.device != device
        for tensor in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kg_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    block_k = triton.next_power_of_2(key_dim)
    # Grid specialization: large tiles halve redundant K-side loads and win once
    # enough programs keep the NPU busy; below that, smaller tiles keep more
    # programs in flight. Measured crossover: 96 programs (batch*heads*V tiles).
    is_a5 = is_950()
    block_v = min(triton.next_power_of_2(value_dim), RECOVERSSM_VALUE_TILE)
    if is_a5 and block_v < RECOVERSSM_VALUE_TILE_LARGE and batch * num_heads >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
        large_v = min(triton.next_power_of_2(value_dim), RECOVERSSM_VALUE_TILE_LARGE)
        if batch * num_heads * triton.cdiv(value_dim, large_v) >= RECOVERSSM_LARGE_TILE_MIN_PROGRAMS:
            block_v = large_v
    grid = (triton.cdiv(value_dim, block_v), num_heads, batch)
    _recoverssm_verify_kernel[grid](
        # Inputs, persistent checkpoint, and per-token records.
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        gate_ptr=raw_g,
        beta_ptr=raw_beta,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        state_ptr=checkpoint_state,
        correction_cache_ptr=correction_cache,
        kg_cache_ptr=kg_cache,
        gate_cache_ptr=None,
        query_start_loc_ptr=query_start_loc,
        state_indices_ptr=state_indices,
        out_ptr=out,
        lower_bound=lower_bound or 0.0,
        null_block_id=NULL_BLOCK_ID,
        # Activation and checkpoint layouts.
        stride_q_token=q.stride(1),
        stride_q_head=q.stride(2),
        stride_k_token=k.stride(1),
        stride_k_head=k.stride(2),
        stride_v_token=v.stride(1),
        stride_v_head=v.stride(2),
        stride_gate_token=raw_g.stride(1),
        stride_gate_head=raw_g.stride(2),
        stride_beta_token=raw_beta.stride(1),
        stride_beta_head=raw_beta.stride(2),
        state_strides=checkpoint_state.stride(),
        correction_strides=correction_cache.stride(),
        kg_cache_strides=kg_cache.stride(),
        # KDA stores raw key/gate together in kg_cache_ptr. The standalone
        # gate page and its strides are removed by KDA_MODE specialization.
        gate_cache_strides=(0, 0, 0),
        stride_out_token=out.stride(1),
        stride_out_head=out.stride(2),
        stride_query_start_loc=query_start_loc.stride(0),
        stride_state_indices=state_indices.stride(0),
        # Compile-time shape and input-format specialization.
        NUM_KEY_HEADS=num_heads,
        NUM_VALUE_HEADS=num_heads,
        K=key_dim,
        V=value_dim,
        BK=block_k,
        BV=block_v,
        INPUT_MODE=KDA_MODE,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


def _conv_compaction_block_d(dim_stride):
    # A3 strided gathers need smaller tiles to fit the 192 KiB UB.
    return RECOVERSSM_CONV_TILE if is_950() or dim_stride == 1 else RECOVERSSM_CONV_STRIDED_TILE


@dataclass
class KDARecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    correction_caches: tuple[torch.Tensor, ...]
    correction_cache_base_addrs: torch.Tensor
    correction_cache_block_strides: torch.Tensor
    kg_caches: tuple[torch.Tensor, ...]
    kg_cache_base_addrs: torch.Tensor
    kg_cache_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    state_commit_lens: torch.Tensor
    boundary_commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float | None
    spec_query_len: int
    conv_dim_stride: int | None

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
        backend: Literal["cute", "triton"] = "triton",
    ) -> "KDARecoverSSMCommitContext":
        """Bind layer-owned page views and allocate reusable commit-plan buffers.

        ``backend`` is accepted for signature compatibility with the upstream
        factory on vLLM main, which selects between its CuTe and Triton commit
        kernels. Ascend only implements the Triton plan, so the value is
        ignored.
        """
        del backend
        if not layers:
            raise ValueError("KDA RecoverSSM commit requires at least one layer")
        if any(len(layer.kv_cache) != 4 for layer in layers):
            raise ValueError("KDA RecoverSSM pages must contain conv, state, correction, and key/gate")
        init_device_properties_triton()

        conv_states = [layer.kv_cache[0] for layer in layers]
        if not is_conv_state_dim_first():
            conv_states = [state.transpose(-1, -2) for state in conv_states]
        checkpoints = [layer.kv_cache[1] for layer in layers]
        correction_caches = [layer.kv_cache[2] for layer in layers]
        kg_caches = [layer.kv_cache[3] for layer in layers]
        A_log = [layer.A_log.reshape(-1) for layer in layers]
        dt_bias = [layer.dt_bias.view(layer.local_num_heads, layer.head_dim) for layer in layers]
        lower_bounds = {layer.gate_lower_bound for layer in layers}
        if len(lower_bounds) != 1:
            raise ValueError("KDA RecoverSSM layers need matching gate bounds")

        state_ref = checkpoints[0]
        if state_ref.ndim != 4:
            raise ValueError("KDA RecoverSSM checkpoint must be four-dimensional")
        num_blocks, num_heads, value_dim, key_dim = state_ref.shape
        for state in checkpoints:
            if (
                state.shape != state_ref.shape
                or state.dtype != state_ref.dtype
                or state.device != state_ref.device
                or state.stride()[1:] != state_ref.stride()[1:]
            ):
                raise ValueError("KDA RecoverSSM layers need matching checkpoint layout")
        expected_correction_shape = (
            num_blocks,
            num_heads,
            spec_query_len,
            value_dim,
        )
        correction_ref = correction_caches[0]
        for correction_cache in correction_caches:
            if (
                correction_cache.shape != expected_correction_shape
                or correction_cache.dtype != torch.float32
                or correction_cache.device != state_ref.device
                or correction_cache.stride()[1:] != correction_ref.stride()[1:]
            ):
                raise ValueError(f"KDA RecoverSSM correction buffers need float32 shape {expected_correction_shape}")
        expected_kg_shape = (num_blocks, num_heads, spec_query_len, 2 * key_dim)
        kg_ref = kg_caches[0]
        for kg_cache in kg_caches:
            if (
                kg_cache.shape != expected_kg_shape
                or kg_cache.dtype != kg_ref.dtype
                or kg_cache.device != state_ref.device
                or kg_cache.stride()[1:] != kg_ref.stride()[1:]
            ):
                raise ValueError(f"KDA RecoverSSM key/gate buffers need shape {expected_kg_shape}")
        if any(param.shape != (num_heads,) for param in A_log):
            raise ValueError("KDA RecoverSSM A_log shape is incompatible")
        if any(param.shape != (num_heads, key_dim) for param in dt_bias):
            raise ValueError("KDA RecoverSSM dt_bias shape is incompatible")

        conv_ref = conv_states[0]
        if conv_ref.ndim != 3:
            raise ValueError("KDA RecoverSSM conv state must be three-dimensional")
        conv_dim, conv_state_len = conv_ref.shape[1:]
        conv_history_len = conv_state_len - spec_query_len + 1
        if conv_history_len <= 0:
            raise ValueError("KDA RecoverSSM conv state is shorter than its window")
        for conv_state in conv_states:
            if (
                conv_state.shape != conv_ref.shape
                or conv_state.dtype != conv_ref.dtype
                or conv_state.device != state_ref.device
                or conv_state.shape[0] != num_blocks
            ):
                raise ValueError("KDA RecoverSSM layers need matching conv state")

        device = state_ref.device

        def _base_addrs(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.data_ptr() for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        def _block_strides(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.tensor(
                [tensor.stride(0) for tensor in tensors],
                dtype=torch.int64,
                device=device,
            )

        return cls(
            conv_states=tuple(conv_states),
            conv_state_base_addrs=_base_addrs(conv_states),
            conv_state_block_strides=_block_strides(conv_states),
            conv_state_dim_strides=torch.tensor(
                [state.stride(1) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_state_token_strides=torch.tensor(
                [state.stride(2) for state in conv_states],
                dtype=torch.int64,
                device=device,
            ),
            conv_history_len=conv_history_len,
            checkpoints=tuple(checkpoints),
            state_base_addrs=_base_addrs(checkpoints),
            state_block_strides=_block_strides(checkpoints),
            correction_caches=tuple(correction_caches),
            correction_cache_base_addrs=_base_addrs(correction_caches),
            correction_cache_block_strides=_block_strides(correction_caches),
            kg_caches=tuple(kg_caches),
            kg_cache_base_addrs=_base_addrs(kg_caches),
            kg_cache_block_strides=_block_strides(kg_caches),
            commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            state_commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            boundary_commit_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            final_state_indices=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            boundary_state_indices=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            boundary_recovery_lens=torch.empty(max_num_reqs, dtype=torch.int32, device=device),
            A_log=torch.stack(tuple(A_log)).contiguous(),
            dt_bias=torch.stack(tuple(dt_bias)).contiguous(),
            lower_bound=lower_bounds.pop(),
            spec_query_len=spec_query_len,
            conv_dim_stride=(
                conv_ref.stride(1) if all(state.stride(1) == conv_ref.stride(1) for state in conv_states) else None
            ),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
    ) -> None:
        """Fold accepted KDA and convolution inputs into every layer."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("KDA RecoverSSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(arg is None for arg in align_args):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError("KDA RecoverSSM align block size must cover one speculative window")
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("KDA RecoverSSM block table must be two-dimensional")
        device = self.checkpoints[0].device
        if (
            any(
                tensor.device != device
                for tensor in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                )
            )
            or (request_indices is not None and request_indices.device != device)
            or (block_table is not None and block_table.device != device)
            or (num_computed_tokens is not None and num_computed_tokens.device != device)
        ):
            raise ValueError("KDA RecoverSSM commit inputs must be on the same device")

        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = 0 if num_computed_tokens is None else num_computed_tokens.stride(0)

        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        block_history = triton.next_power_of_2(self.conv_history_len)
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
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_stride,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        conv_block_d = _conv_compaction_block_d(self.conv_dim_stride)
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, conv_block_d), batch, num_layers)](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=conv_block_d,
            BLOCK_HISTORY=block_history,
            ALIGN_MODE=block_table is not None,
            DIM_STRIDE=self.conv_dim_stride,
            USE_1D_CHANNEL_GATHER=not is_950(),
            num_warps=4,
        )

        state_ref = self.checkpoints[0]
        correction_ref = self.correction_caches[0]
        assert correction_ref.dtype == torch.float32
        _, num_heads, value_dim, key_dim = state_ref.shape
        block_k = triton.next_power_of_2(key_dim)
        matrix_element_size = correction_ref.element_size()
        is_a5 = is_950()
        align_mode = block_table is not None
        # Most aligned decode steps do not cross a cache-block boundary. Recover
        # them without boundary matrices; only crossing requests use the full path.
        # Both kernels select requests on-device, without synchronizing graph replay.
        grouped_commit = (
            is_a5 and state_ref.dtype == torch.float32 and key_dim == value_dim == 128 and num_heads in (32, 64)
        )
        value_tile = RECOVERSSM_VALUE_TILE
        if grouped_commit:
            head_groups = num_heads // RECOVERSSM_COMMIT_HEAD_TILE
            commit_kernel = _commit_recoverssm_state_grouped_kernel
            commit_options = {}
        else:
            if is_a5:
                value_tile *= 2
            # Runtime iteration keeps accumulator/merge matrices independent of window length.
            bytes_per_row = RECOVERSSM_COMMIT_UB_MATRICES * block_k * matrix_element_size
            # Reserve FP32 vector/temporary workspace in addition to matrices.
            max_rows = max(1, get_ub_size_bytes() // bytes_per_row - RECOVERSSM_COMMIT_UB_HEADROOM_ROWS)
            value_tile = min(value_tile, 1 << (max_rows.bit_length() - 1))
            head_groups = num_heads
            commit_kernel = _commit_recoverssm_state_kernel
            commit_options = {"UNROLL_COMMIT": is_a5}
        block_v = min(triton.next_power_of_2(value_dim), value_tile)
        # Explicit flat indexing keeps request rows outermost on both 3.2/3.6
        # backends, balancing mixed boundary requests without a new static flag.
        grid = (triton.cdiv(value_dim, block_v) * num_layers * head_groups * batch,)
        commit_prefix_args = (
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            correction_ref,
            self.correction_cache_base_addrs,
            self.correction_cache_block_strides,
            self.kg_caches[0],
            self.kg_cache_base_addrs,
            self.kg_cache_block_strides,
            self.A_log,
            self.dt_bias,
            state_indices,
        )
        commit_suffix_args = (
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.lower_bound or 0.0,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            state_ref.stride(3),
            correction_ref.stride(1),
            correction_ref.stride(2),
            correction_ref.stride(3),
            self.kg_caches[0].stride(1),
            self.kg_caches[0].stride(2),
            self.kg_caches[0].stride(3),
            self.A_log.stride(0),
            self.A_log.stride(1),
            self.dt_bias.stride(0),
            self.dt_bias.stride(1),
            self.dt_bias.stride(2),
            state_indices.stride(0),
            num_layers,
        )
        commit_meta = dict(
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=num_heads,
            USE_LOWER_BOUND=self.lower_bound is not None,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=4,
            num_stages=2,
        )
        commit_kernel[grid](
            *commit_prefix_args,
            self.state_commit_lens if align_mode else self.commit_lens,
            *commit_suffix_args,
            **commit_meta,
            ALIGN_MODE=False,
            **commit_options,
        )
        if align_mode:
            # Boundary snapshots need six matrices per head.
            bytes_per_row = RECOVERSSM_BOUNDARY_COMMIT_UB_MATRICES * block_k * matrix_element_size
            if grouped_commit:
                bytes_per_row *= RECOVERSSM_COMMIT_HEAD_TILE
            max_rows = max(1, get_ub_size_bytes() // bytes_per_row - RECOVERSSM_COMMIT_UB_HEADROOM_ROWS)
            boundary_tile = min(RECOVERSSM_VALUE_TILE, 1 << (max_rows.bit_length() - 1))
            boundary_block_v = min(triton.next_power_of_2(value_dim), boundary_tile)
            boundary_meta = {**commit_meta, "BV": boundary_block_v}
            commit_kernel[(triton.cdiv(value_dim, boundary_block_v) * num_layers * head_groups * batch,)](
                *commit_prefix_args,
                self.boundary_commit_lens,
                *commit_suffix_args,
                **boundary_meta,
                ALIGN_MODE=True,
                **commit_options,
            )


__all__ = ["KDARecoverSSMCommitContext", "kda_recoverssm_verify"]
