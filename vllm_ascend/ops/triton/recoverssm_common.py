# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared RecoverSSM verification and commit primitives for recurrent attention."""

from vllm.triton_utils import tl, triton

KDA_MODE: tl.constexpr = 0
GDN_MODE: tl.constexpr = 1


@triton.jit
def _l2_normalize_vector(vector):
    # AscendC normalizes nonzero vectors without CUDA's squared-norm bias.
    norm_square = tl.sum(vector * vector)
    norm_input = tl.where(norm_square > 0.0, norm_square, 1.0)
    tl.static_assert(norm_input.dtype == tl.float32)
    inverse = tl.rsqrt(norm_input)
    return tl.where(norm_square > 0.0, vector * inverse, 0.0)


@triton.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: tl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        scaled_gate = A * gate_input
        tl.static_assert(scaled_gate.dtype == tl.float32)
        return lower_bound * tl.sigmoid(scaled_gate)
    tl.static_assert(gate_input.dtype == tl.float32)
    softplus_input = 1.0 + tl.exp(gate_input)
    tl.static_assert(softplus_input.dtype == tl.float32)
    log_softplus = tl.log(softplus_input)
    softplus_gate = tl.where(gate_input > 20.0, gate_input, log_softplus)
    return -A * softplus_gate


@triton.jit
def _copy_conv_state_tile(
    source_ptr,
    final_ptr,
    boundary_ptr,
    channel_offsets,
    history_offsets,
    token_stride,
    commit_len,
    boundary_recovery_len,
    mask,
    boundary_valid,
    ALIGN_MODE: tl.constexpr,
):
    final_values = tl.load(
        source_ptr + channel_offsets + (commit_len - 1 + history_offsets) * token_stride,
        mask=mask,
    )
    if ALIGN_MODE:
        boundary_values = tl.load(
            source_ptr + channel_offsets + (boundary_recovery_len - 1 + history_offsets) * token_stride,
            mask=mask & boundary_valid,
        )
        tl.store(
            boundary_ptr + channel_offsets + history_offsets * token_stride,
            boundary_values,
            mask=mask & boundary_valid,
        )
    tl.store(
        final_ptr + channel_offsets + history_offsets * token_stride,
        final_values,
        mask=mask,
    )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    state_commit_lens_ptr,
    boundary_commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    INPUT_MODE: tl.constexpr = KDA_MODE,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(tl.int64)
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(request_indices_ptr + spec_idx * stride_request_indices)
        if INPUT_MODE == KDA_MODE:
            request_idx = request_idx.to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted)
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc)
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(tl.int32)
        final_num_computed = num_computed + commit_len
        final_state_col = tl.minimum(final_num_computed // mamba_block_size, block_table_width - 1)
        final_state_idx = tl.load(
            block_table_ptr + request_idx * stride_block_table_row + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    has_boundary = (boundary_state_idx > null_block_id) & (boundary_recovery_len > 0)
    # Predicate once per request, rather than reading several metadata arrays in
    # every head/value program merely to discover that it has no work.
    tl.store(state_commit_lens_ptr + spec_idx, tl.where(valid & ~has_boundary, commit_len, 0))
    tl.store(boundary_commit_lens_ptr + spec_idx, tl.where(valid & has_boundary, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


@triton.jit
def _compact_conv_state_kernel(
    conv_state_ref_ptr,
    conv_state_base_addrs_ptr,
    conv_state_block_strides_ptr,
    conv_state_dim_strides_ptr,
    conv_state_token_strides_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    conv_dim,
    conv_history_len,
    stride_state_indices,
    BLOCK_D: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    DIM_STRIDE: tl.constexpr = None,
    USE_1D_CHANNEL_GATHER: tl.constexpr = False,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_l = tl.program_id(2)
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    base_addr = tl.load(conv_state_base_addrs_ptr + pid_l)
    block_stride = tl.load(conv_state_block_strides_ptr + pid_l)
    if DIM_STRIDE is None:
        dim_stride = tl.load(conv_state_dim_strides_ptr + pid_l)
    else:
        dim_stride = DIM_STRIDE
    token_stride = tl.load(conv_state_token_strides_ptr + pid_l)
    conv_state_ptr = base_addr.to(tl.pointer_type(conv_state_ref_ptr.dtype.element_ty))
    source_ptr = conv_state_ptr + source_state_idx * block_stride
    final_ptr = conv_state_ptr + final_state_idx * block_stride

    boundary_ptr = None
    if ALIGN_MODE:
        boundary_ptr = conv_state_ptr + boundary_state_idx * block_stride

    if USE_1D_CHANNEL_GATHER and (DIM_STRIDE is not None and DIM_STRIDE != 1):
        # A3 Triton 3.2 requires one-dimensional gathers for strided channels.
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        for history in tl.static_range(BLOCK_HISTORY):
            mask = (offs_d < conv_dim) & (history < conv_history_len)
            _copy_conv_state_tile(
                source_ptr,
                final_ptr,
                boundary_ptr,
                offs_d * dim_stride,
                history,
                token_stride,
                commit_len,
                boundary_recovery_len,
                mask,
                boundary_state_idx > null_block_id,
                ALIGN_MODE,
            )
        return
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]
    offs_h = tl.arange(0, BLOCK_HISTORY)[:, None]
    mask = (offs_d < conv_dim) & (offs_h < conv_history_len)
    _copy_conv_state_tile(
        source_ptr,
        final_ptr,
        boundary_ptr,
        offs_d * dim_stride,
        offs_h,
        token_stride,
        commit_len,
        boundary_recovery_len,
        mask,
        boundary_state_idx > null_block_id,
        ALIGN_MODE,
    )


@triton.jit
def _load_recoverssm_commit_record(
    correction_ptr,
    kg_ptr,
    gate_ptr,
    offs_k,
    offs_v,
    mask_k,
    mask_v,
    stride_correction_dim,
    stride_kg_dim,
    A,
    dt_bias,
    lower_bound,
    K: tl.constexpr,
    INPUT_MODE: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    GROUPED: tl.constexpr = False,
):
    # Pointers already select the layer, block, head and token record.
    # Keep GDN's decay scalar; KDA decays individual key channels.
    k = tl.load(
        kg_ptr + offs_k * stride_kg_dim,
        mask=mask_k,
        other=0.0 if mask_k is not None else None,
    ).to(tl.float32)
    correction = tl.load(
        correction_ptr + offs_v * stride_correction_dim,
        mask=mask_v,
        other=0.0 if mask_v is not None else None,
    ).to(tl.float32)
    if INPUT_MODE == KDA_MODE:
        raw_g = tl.load(
            kg_ptr + (K + offs_k) * stride_kg_dim,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
        if GROUPED:
            normalized_k = _l2_normalize_per_head(k)
        else:
            normalized_k = _l2_normalize_vector(k)
        gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
        tl.static_assert(gate.dtype == tl.float32)
    else:
        normalized_k = k
        gate = tl.load(gate_ptr)
    return correction, normalized_k, tl.exp(gate)


@triton.jit
def _recoverssm_recurrent_update(
    state,
    normalized_k,
    v,
    gate,
    beta,
    need_state_update,
    INPUT_MODE: tl.constexpr,
):
    decay = tl.exp(gate)
    if INPUT_MODE == KDA_MODE:
        # Preserve KDA's original multiply order: key is weighted before the
        # state-key reduction, rather than weighting the state first.
        correction = (v - tl.sum(state * (normalized_k * decay)[None, :], axis=1)) * beta
        # The last KDA token only needs output and compact records.
        # Its updated local state has no consumer.
        next_state = state
        if need_state_update:
            next_state = state * decay[None, :] + correction[:, None] * normalized_k[None, :]
    else:
        decayed_state = state * decay[None, :]
        correction = (v - tl.sum(decayed_state * normalized_k[None, :], axis=1)) * beta
        # GDN output consumes next_state, including for the last token.
        next_state = decayed_state + correction[:, None] * normalized_k[None, :]
    return decay, next_state, correction


@triton.jit
def _recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kg_cache_ptr,
    gate_cache_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    out_ptr,
    lower_bound,
    null_block_id,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_k_token: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_v_token: tl.constexpr,
    stride_v_head: tl.constexpr,
    stride_gate_token: tl.constexpr,
    stride_gate_head: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_beta_head: tl.constexpr,
    state_strides: tl.constexpr,
    correction_strides: tl.constexpr,
    kg_cache_strides: tl.constexpr,
    gate_cache_strides: tl.constexpr,
    stride_out_token: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_query_start_loc: tl.constexpr,
    stride_state_indices: tl.constexpr,
    NUM_KEY_HEADS: tl.constexpr,
    NUM_VALUE_HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    INPUT_MODE: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    # Preserve physical page layouts, including padding between blocks or heads.
    # State: [block, head, value, key]; records: [block, head, pos, dim].
    stride_state_block: tl.constexpr = state_strides[0]
    stride_state_head: tl.constexpr = state_strides[1]
    stride_state_v: tl.constexpr = state_strides[2]
    stride_state_k: tl.constexpr = state_strides[3]
    stride_correction_block: tl.constexpr = correction_strides[0]
    stride_correction_head: tl.constexpr = correction_strides[1]
    stride_correction_position: tl.constexpr = correction_strides[2]
    stride_correction_dim: tl.constexpr = correction_strides[3]
    stride_kg_cache_block: tl.constexpr = kg_cache_strides[0]
    stride_kg_cache_head: tl.constexpr = kg_cache_strides[1]
    stride_kg_cache_position: tl.constexpr = kg_cache_strides[2]
    stride_kg_cache_dim: tl.constexpr = kg_cache_strides[3]
    if INPUT_MODE == GDN_MODE:
        # Only GDN has a separate scalar-gate page: [block, head, pos].
        stride_gate_cache_block: tl.constexpr = gate_cache_strides[0]
        stride_gate_cache_head: tl.constexpr = gate_cache_strides[1]
        stride_gate_cache_pos: tl.constexpr = gate_cache_strides[2]

    # Advance a register-held checkpoint and record each token's compact
    # correction/key/gate; persistent checkpoint pages remain read-only.
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(2)
    pid_h = tl.program_id(1)

    bos = tl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(tl.int64)
    if bos == eos:
        return
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    # Provably-full tiles (K a power of two, BV dividing V) drop their
    # predication masks entirely; ragged shapes keep the original masks.
    if K == BK:
        mask_k = None
    else:
        mask_k = offs_k < K
    if V % BV == 0:
        mask_v = None
    else:
        mask_v = offs_v < V
    if (K == BK) and (V % BV == 0):
        mask_state = None
    elif K == BK:
        mask_state = tl.broadcast_to((offs_v < V)[:, None], (BV, BK))
    elif V % BV == 0:
        mask_state = tl.broadcast_to((offs_k < K)[None, :], (BV, BK))
    else:
        mask_state = (offs_v < V)[:, None] & (offs_k < K)[None, :]

    if state_idx <= null_block_id:
        for token_offset in range(query_len):
            tl.store(out_ptr + (bos + token_offset) * stride_out_token + pid_h * stride_out_head + offs_v, 0.0, mask_v)
        return

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)

    if INPUT_MODE == KDA_MODE: # 这个判断是必要的吗？
        a_log = tl.load(A_log_ptr + pid_h).to(tl.float32)
        tl.static_assert(a_log.dtype == tl.float32)
        A = tl.exp(a_log)
        dt_bias = tl.load(
            dt_bias_ptr + pid_h * K + offs_k,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)

    key_head = pid_h // (NUM_VALUE_HEADS // NUM_KEY_HEADS)
    # token_offset < query_len always holds inside range(query_len); the
    # token_valid masks and the tl.where state select are redundant and dropped.
    for token_offset in range(query_len):
        token = bos + token_offset
        q = tl.load(
            q_ptr + token * stride_q_token + key_head * stride_q_head + offs_k,
            mask_k,
            0.0 if mask_k is not None else None,
        ).to(tl.float32)
        k = tl.load(
            k_ptr + token * stride_k_token + key_head * stride_k_head + offs_k,
            mask_k,
            0.0 if mask_k is not None else None,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + token * stride_v_token + pid_h * stride_v_head + offs_v,
            mask_v,
            0.0 if mask_v is not None else None,
        ).to(tl.float32)
        beta = tl.load(beta_ptr + token * stride_beta_token + pid_h * stride_beta_head).to(tl.float32)

        if INPUT_MODE == KDA_MODE:
            raw_g = tl.load(
                gate_ptr + token * stride_gate_token + pid_h * stride_gate_head + offs_k,
                mask_k,
                0.0 if mask_k is not None else None,
            ).to(tl.float32)
            raw_beta = beta
            q = _l2_normalize_vector(q) * (K**-0.5)
            normalized_k = _l2_normalize_vector(k)
            gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)

            tl.static_assert(raw_beta.dtype == tl.float32)
            beta = tl.sigmoid(raw_beta)
        else:
            # GDN supplies an effective scalar log-decay. Broadcasting it to
            # BK lets the recurrent primitive share KDA's vector-gate shape.
            effective_gate = tl.load(gate_ptr + token * stride_gate_token + pid_h * stride_gate_head).to(tl.float32)
            gate = effective_gate + tl.zeros([BK], dtype=tl.float32)
            normalized_k = k
        need_state_update = token_offset < query_len - 1
        decay, next_state, correction = _recoverssm_recurrent_update(
            state,
            normalized_k,
            v,
            gate,
            beta,
            need_state_update,
            INPUT_MODE,
        )
        if INPUT_MODE == KDA_MODE:
            # KDA scales the normalized query before both reductions. Keep this
            # expanded form to preserve the existing verify kernel's FP ordering.
            out = tl.sum(state * (q * decay)[None, :], axis=1) + correction * tl.sum(normalized_k * q)
        else:
            out = tl.sum(next_state * q[None, :], axis=1) * (K ** -0.5)
        tl.store(out_ptr + token * stride_out_token + pid_h * stride_out_head + offs_v, out, mask_v)
        # Update the state for next loop.
        state = next_state

        correction_ptr = (
            correction_cache_ptr
            + state_idx * stride_correction_block
            + pid_h * stride_correction_head
            + token_offset * stride_correction_position
        )
        tl.store(
            correction_ptr + offs_v * stride_correction_dim,
            correction,
            mask_v,
        )

        if pid_v == 0:
            if INPUT_MODE == KDA_MODE:
                kg_ptr = (
                    kg_cache_ptr
                    + state_idx * stride_kg_cache_block
                    + pid_h * stride_kg_cache_head
                    + token_offset * stride_kg_cache_position
                )
                tl.store(kg_ptr + offs_k * stride_kg_cache_dim, k, mask_k)
                tl.store(kg_ptr + (K + offs_k) * stride_kg_cache_dim, raw_g, mask_k)
            else:
                gate_record = (
                    gate_cache_ptr
                    + state_idx * stride_gate_cache_block
                    + pid_h * stride_gate_cache_head
                    + token_offset * stride_gate_cache_pos
                )
                tl.store(gate_record, effective_gate)
                if pid_h % (NUM_VALUE_HEADS // NUM_KEY_HEADS) == 0:
                    tl.store(
                        kg_cache_ptr
                        + state_idx * stride_kg_cache_block
                        + key_head * stride_kg_cache_head
                        + token_offset * stride_kg_cache_position
                        + offs_k * stride_kg_cache_dim,
                        k,
                        mask_k,
                    )


@triton.jit
def _accumulate_recoverssm_commit_step(
    correction_cache_ptr,
    kg_cache_ptr,
    token_offset,
    offs_k,
    offs_v,
    mask_k,
    mask_v,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    A,
    dt_bias,
    lower_bound,
    boundary_recovery_len,
    final_decay,
    final_correction,
    boundary_decay,
    boundary_correction,
    K: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    gate_cache_ptr,
    stride_gate_cache_pos,
    INPUT_MODE: tl.constexpr,
):
    correction_ptr = correction_cache_ptr + token_offset * stride_correction_cache_pos
    kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
    gate_ptr = None
    if INPUT_MODE == GDN_MODE:
        gate_ptr = gate_cache_ptr + token_offset * stride_gate_cache_pos
    correction, normalized_k, decay = _load_recoverssm_commit_record(
        correction_ptr=correction_ptr,
        kg_ptr=kg_ptr,
        gate_ptr=gate_ptr,
        offs_k=offs_k,
        offs_v=offs_v,
        mask_k=mask_k,
        mask_v=mask_v,
        stride_correction_dim=stride_correction_cache_dim,
        stride_kg_dim=stride_kg_cache_dim,
        A=A,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        K=K,
        INPUT_MODE=INPUT_MODE,
        USE_LOWER_BOUND=USE_LOWER_BOUND,
    )
    weighted_k = normalized_k * final_decay
    final_correction = tl.fma(correction[:, None], weighted_k[None, :], final_correction)
    final_decay *= decay
    if ALIGN_MODE:
        before_boundary = token_offset < boundary_recovery_len
        boundary_k = normalized_k * boundary_decay
        boundary_correction = tl.where(
            before_boundary,
            tl.fma(correction[:, None], boundary_k[None, :], boundary_correction),
            boundary_correction,
        )
        boundary_decay *= tl.where(before_boundary, decay, 1.0)
    return final_decay, final_correction, boundary_decay, boundary_correction


@triton.jit(do_not_specialize=["num_layers"])
def _commit_recoverssm_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    num_layers,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    UNROLL_COMMIT: tl.constexpr,
    # Add below params for GDN MODE
    gate_cache_base_addrs_ptr=None,
    gate_cache_block_strides_ptr=None,
    stride_gate_cache_head=0,
    stride_gate_cache_pos=0,
    NUM_KEY_HEADS: tl.constexpr = 0,
    INPUT_MODE: tl.constexpr = KDA_MODE,
):
    # Explicit one-dimensional indexing gives the same request-major order
    # across Ascend Triton backends with different 3-D grid flattening orders.
    pid = tl.program_id(0)
    value_tiles = tl.cdiv(V, BV)
    pid_v = pid % value_tiles
    pid_lh = pid // value_tiles
    layer_head_tiles = num_layers * NUM_HEADS
    pid_b = pid_lh // layer_head_tiles
    pid_lh %= layer_head_tiles
    # Walk accepted records backwards, composing decay-weighted rank-one
    # updates before combining them with the unchanged initial checkpoint.
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)

    if final_state_idx <= null_block_id:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head

    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty))
    correction_cache_ptr += source_state_idx * correction_cache_block_stride + pid_h * stride_correction_cache_head
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty))
    pid_kh = pid_h
    gate_cache_ptr = None
    if INPUT_MODE == GDN_MODE:
        tl.static_assert(NUM_KEY_HEADS > 0 and NUM_HEADS % NUM_KEY_HEADS == 0)
        pid_kh = pid_h // (NUM_HEADS // NUM_KEY_HEADS)
        gate_base_addr = tl.load(gate_cache_base_addrs_ptr + pid_l)
        gate_block_stride = tl.load(gate_cache_block_strides_ptr + pid_l)
        gate_cache_ptr = gate_base_addr.to(tl.pointer_type(tl.float32))
        gate_cache_ptr += source_state_idx * gate_block_stride + pid_h * stride_gate_cache_head
    kg_cache_ptr += source_state_idx * kg_cache_block_stride + pid_kh * stride_kg_cache_head

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = None if K == BK else offs_k < K
    mask_v = None if V % BV == 0 else offs_v < V
    if K == BK and V % BV == 0:
        mask_state = None
    elif K == BK:
        mask_state = tl.broadcast_to((offs_v < V)[:, None], (BV, BK))
    elif V % BV == 0:
        mask_state = tl.broadcast_to((offs_k < K)[None, :], (BV, BK))
    else:
        mask_state = (offs_v < V)[:, None] & (offs_k < K)[None, :]
    state_ptrs = source_state_ptr + offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k
    A = None
    dt_bias = None
    if INPUT_MODE == KDA_MODE:
        a_log = tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(tl.float32)
        tl.static_assert(a_log.dtype == tl.float32)
        A = tl.exp(a_log)

        dt_bias = tl.load(
            dt_bias_ptr + pid_l * stride_dt_bias_layer + pid_h * stride_dt_bias_head + offs_k * stride_dt_bias_dim,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
    final_decay = tl.full([BK], 1.0, tl.float32)
    final_correction = tl.zeros([BV, BK], tl.float32)
    boundary_decay = tl.full([BK], 1.0, tl.float32)
    boundary_correction = tl.zeros([BV, BK], tl.float32)

    # A3 runtime iteration bounds UB use; A5 retains its faster static expansion.
    if UNROLL_COMMIT:
        for reverse_offset in tl.static_range(SPEC_QUERY_LEN):
            if reverse_offset < commit_len:
                final_decay, final_correction, boundary_decay, boundary_correction = _accumulate_recoverssm_commit_step(
                    correction_cache_ptr,
                    kg_cache_ptr,
                    commit_len - reverse_offset - 1,
                    offs_k,
                    offs_v,
                    mask_k,
                    mask_v,
                    stride_correction_cache_pos,
                    stride_correction_cache_dim,
                    stride_kg_cache_pos,
                    stride_kg_cache_dim,
                    A,
                    dt_bias,
                    lower_bound,
                    boundary_recovery_len,
                    final_decay,
                    final_correction,
                    boundary_decay,
                    boundary_correction,
                    K,
                    USE_LOWER_BOUND,
                    ALIGN_MODE,
                    gate_cache_ptr,
                    stride_gate_cache_pos,
                    INPUT_MODE,
                )
    else:
        for reverse_offset in range(tl.minimum(commit_len, SPEC_QUERY_LEN)):
            if reverse_offset < commit_len:
                final_decay, final_correction, boundary_decay, boundary_correction = _accumulate_recoverssm_commit_step(
                    correction_cache_ptr,
                    kg_cache_ptr,
                    commit_len - reverse_offset - 1,
                    offs_k,
                    offs_v,
                    mask_k,
                    mask_v,
                    stride_correction_cache_pos,
                    stride_correction_cache_dim,
                    stride_kg_cache_pos,
                    stride_kg_cache_dim,
                    A,
                    dt_bias,
                    lower_bound,
                    boundary_recovery_len,
                    final_decay,
                    final_correction,
                    boundary_decay,
                    boundary_correction,
                    K,
                    USE_LOWER_BOUND,
                    ALIGN_MODE,
                    gate_cache_ptr,
                    stride_gate_cache_pos,
                    INPUT_MODE,
                )
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    state = initial_state * final_decay[None, :] + final_correction
    if ALIGN_MODE:
        boundary_ptrs = (
            state_ptr
            + boundary_state_idx * state_block_stride
            + pid_h * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )
        boundary_state = initial_state * boundary_decay[None, :] + boundary_correction
        tl.store(
            boundary_ptrs,
            boundary_state,
            mask=(boundary_state_idx > null_block_id)
            if mask_state is None
            else mask_state & (boundary_state_idx > null_block_id),
        )

    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


@triton.jit
def _l2_normalize_per_head(vector):
    norm_square = tl.sum(vector * vector, axis=1)
    norm_input = tl.where(norm_square > 0.0, norm_square, 1.0)
    tl.static_assert(norm_input.dtype == tl.float32)
    inverse = tl.rsqrt(norm_input)
    return tl.where(norm_square[:, None] > 0.0, vector * inverse[:, None], 0.0)


@triton.jit(do_not_specialize=["num_layers"])
def _commit_recoverssm_state_grouped_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_strides_ptr,
    kg_cache_ref_ptr,
    kg_cache_base_addrs_ptr,
    kg_cache_block_strides_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    lower_bound,
    null_block_id,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_correction_cache_head,
    stride_correction_cache_pos,
    stride_correction_cache_dim,
    stride_kg_cache_head,
    stride_kg_cache_pos,
    stride_kg_cache_dim,
    stride_A_layer,
    stride_A_head,
    stride_dt_bias_layer,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_state_indices,
    num_layers,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    BH: tl.constexpr = 2,
    # Add below params for GDN MODE
    gate_cache_base_addrs_ptr=None,
    gate_cache_block_strides_ptr=None,
    stride_gate_cache_head=0,
    stride_gate_cache_pos=0,
    NUM_KEY_HEADS: tl.constexpr = 0,
    INPUT_MODE: tl.constexpr = KDA_MODE,
):
    # Commit two independent heads per program.
    tl.static_assert(K == 128)
    tl.static_assert(V == 128)
    tl.static_assert(BK == 128)
    tl.static_assert(BV <= 64)
    pid = tl.program_id(0)
    value_tiles = tl.cdiv(V, BV)
    pid_v = pid % value_tiles
    pid_lh = pid // value_tiles
    layer_head_tiles = num_layers * (NUM_HEADS // BH)
    pid_b = pid_lh // layer_head_tiles
    pid_lh %= layer_head_tiles
    tl.static_assert(NUM_HEADS % BH == 0)
    pid_l = pid_lh // (NUM_HEADS // BH)
    pid_h = pid_lh % (NUM_HEADS // BH) * BH + tl.arange(0, BH)
    commit_len = tl.load(commit_lens_ptr + pid_b)
    if commit_len == 0:
        return
    source_state_idx = tl.load(state_indices_ptr + pid_b * stride_state_indices).to(tl.int64)
    if source_state_idx <= null_block_id:
        return
    final_state_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    if final_state_idx <= null_block_id:
        return
    if ALIGN_MODE:
        boundary_state_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
        boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + pid_b)
    state_base_addr = tl.load(state_base_addrs_ptr + pid_l)
    state_block_stride = tl.load(state_block_strides_ptr + pid_l)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    correction_cache_base_addr = tl.load(correction_cache_base_addrs_ptr + pid_l)
    correction_cache_block_stride = tl.load(correction_cache_block_strides_ptr + pid_l)
    correction_cache_ptr = correction_cache_base_addr.to(tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty))
    correction_cache_ptr += source_state_idx * correction_cache_block_stride + pid_h * stride_correction_cache_head
    kg_cache_base_addr = tl.load(kg_cache_base_addrs_ptr + pid_l)
    kg_cache_block_stride = tl.load(kg_cache_block_strides_ptr + pid_l)
    kg_cache_ptr = kg_cache_base_addr.to(tl.pointer_type(kg_cache_ref_ptr.dtype.element_ty))
    pid_kh = pid_h
    gate_cache_ptr = None
    if INPUT_MODE == GDN_MODE:
        tl.static_assert(NUM_KEY_HEADS > 0 and NUM_HEADS % NUM_KEY_HEADS == 0)
        pid_kh = pid_h // (NUM_HEADS // NUM_KEY_HEADS)
        gate_base_addr = tl.load(gate_cache_base_addrs_ptr + pid_l)
        gate_block_stride = tl.load(gate_cache_block_strides_ptr + pid_l)
        gate_cache_ptr = gate_base_addr.to(tl.pointer_type(tl.float32))
        gate_cache_ptr += source_state_idx * gate_block_stride + pid_h * stride_gate_cache_head
    kg_cache_ptr += source_state_idx * kg_cache_block_stride + pid_kh * stride_kg_cache_head
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = None if K == BK else offs_k < K
    mask_v = None if V % BV == 0 else offs_v < V
    if K == BK and V % BV == 0:
        mask_state = None
    elif K == BK:
        mask_state = tl.broadcast_to((offs_v < V)[:, None], (BV, BK))
    elif V % BV == 0:
        mask_state = tl.broadcast_to((offs_k < K)[None, :], (BV, BK))
    else:
        mask_state = (offs_v < V)[:, None] & (offs_k < K)[None, :]
    if mask_state is not None:
        mask_state = tl.broadcast_to(mask_state[None, :, :], (BH, BV, BK))
    if mask_k is not None:
        mask_k = tl.broadcast_to(mask_k[None, :], (BH, BK))
    if mask_v is not None:
        mask_v = tl.broadcast_to(mask_v[None, :], (BH, BV))
    state_ptrs = (
        source_state_ptr[:, None, None]
        + offs_v[None, :, None] * stride_state_v
        + offs_k[None, None, :] * stride_state_k
    )
    # These two params can is only useful in KDA MODE.
    # In GDN MODE, these two params should be None.
    A = None
    dt_bias = None
    if INPUT_MODE == KDA_MODE:
        a_log = tl.load(A_log_ptr + pid_l * stride_A_layer + pid_h * stride_A_head).to(tl.float32)
        tl.static_assert(a_log.dtype == tl.float32)
        A = tl.exp(a_log)
        dt_bias = tl.load(
            dt_bias_ptr
            + pid_l * stride_dt_bias_layer
            + pid_h[:, None] * stride_dt_bias_head
            + offs_k[None, :] * stride_dt_bias_dim,
            mask=mask_k,
            other=0.0 if mask_k is not None else None,
        ).to(tl.float32)
    final_decay = tl.full([BH, BK], 1.0, tl.float32)
    final_correction = tl.zeros([BH, BV, BK], tl.float32)
    if ALIGN_MODE:
        boundary_decay = tl.full([BH, BK], 1.0, tl.float32)
        boundary_correction = tl.zeros([BH, BV, BK], tl.float32)
    for reverse_offset in tl.static_range(SPEC_QUERY_LEN):
        if reverse_offset < commit_len:
            token_offset = commit_len - reverse_offset - 1
            correction_ptr = correction_cache_ptr + token_offset * stride_correction_cache_pos
            kg_ptr = kg_cache_ptr + token_offset * stride_kg_cache_pos
            gate_ptr = None
            grouped_A = None
            if INPUT_MODE == GDN_MODE:
                gate_ptr = (gate_cache_ptr + token_offset * stride_gate_cache_pos)[:, None]
            else:
                grouped_A = A[:, None]
            correction, normalized_k, decay = _load_recoverssm_commit_record(
                correction_ptr=correction_ptr[:, None],
                kg_ptr=kg_ptr[:, None],
                gate_ptr=gate_ptr,
                offs_k=offs_k[None, :],
                offs_v=offs_v[None, :],
                mask_k=mask_k,
                mask_v=mask_v,
                stride_correction_dim=stride_correction_cache_dim,
                stride_kg_dim=stride_kg_cache_dim,
                A=grouped_A,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                K=K,
                INPUT_MODE=INPUT_MODE,
                USE_LOWER_BOUND=USE_LOWER_BOUND,
                GROUPED=True,
            )
            weighted_k = normalized_k * final_decay
            final_correction = tl.fma(correction[:, :, None], weighted_k[:, None, :], final_correction)
            final_decay *= decay
            if ALIGN_MODE:
                before_boundary = token_offset < boundary_recovery_len
                boundary_k = normalized_k * boundary_decay
                boundary_correction = tl.where(
                    before_boundary,
                    tl.fma(correction[:, :, None], boundary_k[:, None, :], boundary_correction),
                    boundary_correction,
                )
                boundary_decay *= tl.where(before_boundary, decay, 1.0)
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0 if mask_state is not None else None).to(tl.float32)
    state = initial_state * final_decay[:, None, :] + final_correction
    if ALIGN_MODE:
        if boundary_state_idx > null_block_id:
            boundary_ptrs = (
                state_ptr
                + boundary_state_idx * state_block_stride
                + pid_h[:, None, None] * stride_state_head
                + offs_v[None, :, None] * stride_state_v
                + offs_k[None, None, :] * stride_state_k
            )
            boundary_state = initial_state * boundary_decay[:, None, :] + boundary_correction
            tl.store(boundary_ptrs, boundary_state, mask=mask_state)
    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h[:, None, None] * stride_state_head
        + offs_v[None, :, None] * stride_state_v
        + offs_k[None, None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)
