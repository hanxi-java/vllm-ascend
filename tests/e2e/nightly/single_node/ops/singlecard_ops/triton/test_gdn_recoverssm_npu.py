# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify/commit correctness, including grouped heads and paged cache strides."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

import vllm_ascend.ops.triton.gdn_recoverssm as recovery  # noqa: E402


def paged(shape, *, dtype=torch.float32):
    elements = 1
    strides = []
    for size in reversed(shape[1:]):
        strides.append(elements)
        elements *= size
    backing = torch.zeros(shape[0] * (elements + 128), dtype=dtype, device="npu")
    return backing.as_strided(shape, (elements + 128, *reversed(strides)))


def reference(q, k, v, g, beta, initial, starts, indices):
    out = torch.zeros_like(v, dtype=torch.float32)
    states = []
    ratio = v.shape[2] // k.shape[2]
    for row, block in enumerate(indices):
        state = initial[block].float().clone()
        history = [state.clone()]
        for token in range(starts[row], starts[row + 1]):
            key = k[0, token].float().repeat_interleave(ratio, 0)
            query = q[0, token].float().repeat_interleave(ratio, 0)
            state = state * g[0, token].float().exp()[:, None, None]
            residual = (v[0, token].float() - torch.einsum("hvk,hk->hv", state, key)) * beta[0, token].float()[:, None]
            state = state + residual[:, :, None] * key[:, None, :]
            out[0, token] = torch.einsum("hvk,hk->hv", state, query) * (key.shape[-1] ** -0.5)
            history.append(state.clone())
        states.append(history)
    return out, states


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("hk,hv,dk,dv", [(2, 4, 128, 128), (1, 2, 33, 47)])
def test_verify_and_multilayer_commit_preserve_accepted_prefix(monkeypatch, accepted, dim_first, hk, hv, dk, dv):
    torch.manual_seed(73)
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    blocks, window, history = 5, 5, 3
    channels = 2 * hk * dk + hv * dv
    layers = []
    for _ in range(2):
        conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
        conv.copy_(torch.randn_like(conv))
        state = paged((blocks, hv, dv, dk))
        state.copy_(torch.randn_like(state) * 0.1)
        records = (paged((blocks, hv, window, dv)), paged((blocks, hk, window, dk)), paged((blocks, hv, window)))
        layers.append(SimpleNamespace(kv_cache=(conv if dim_first else conv.transpose(-1, -2), state, *records)))
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=4)
    # Reorder physical pages between rounds; sampler rows are in a different order.
    for block_ids in ([3, 1, 0], [1, 3, 0]):
        indices = torch.tensor(block_ids, dtype=torch.int32, device="npu")
        starts = torch.tensor([0, 5, 8, 8], dtype=torch.int32, device="npu")
        counts = [accepted, min(accepted, 3)]
        expected_states, expected_convs = [], []
        for layer, page in zip(layers, context.pages):
            conv, state, corrections, keys, gates = page
            initial, old_conv = state.clone(), conv.clone()
            # Poison previous-round records: rejected/padded entries must not leak.
            for record in (corrections, keys, gates):
                record.fill_(float("nan"))
            q = torch.nn.functional.normalize(torch.randn(1, 8, hk, dk, device="npu"), dim=-1).to(torch.bfloat16)
            k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).to(torch.bfloat16)
            v = torch.randn(1, 8, hv, dv, device="npu", dtype=torch.bfloat16)
            g = -torch.rand(1, 8, hv, device="npu")
            beta = torch.rand(1, 8, hv, device="npu", dtype=torch.bfloat16)
            expected, histories = reference(q, k, v, g, beta, initial, [0, 5, 8, 8], block_ids)
            observed = recovery.gdn_recoverssm_verify(
                q, k, v, g, beta, state, corrections, keys, gates, starts, indices, window
            )
            torch.testing.assert_close(observed.float(), expected, atol=3e-3, rtol=2e-2)
            # RecoverSSM records the effective decay so commit replays the exact
            # value instead of evaluating exp(g) a second time.
            for row, block in enumerate(block_ids):
                begin = starts[row].item()
                length = starts[row + 1].item() - begin
                expected_decay = g[0, begin : begin + length].float().exp().transpose(0, 1)
                torch.testing.assert_close(
                    gates[block, :, :length],
                    expected_decay,
                    atol=2e-3,
                    rtol=2e-3,
                )
            torch.testing.assert_close(state, initial, atol=0, rtol=0)
            final = initial.clone()
            conv_final = old_conv.clone()
            for row, count in enumerate(counts):
                block = block_ids[row]
                final[block] = histories[row][count]
                if count:
                    conv_final[block, :, :history] = old_conv[block, :, count - 1 : count - 1 + history]
            expected_states.append(final)
            expected_convs.append(conv_final)
        sampled = torch.tensor([counts[1], 99, counts[0]], dtype=torch.int32, device="npu")
        mapping = torch.tensor([2, 0], dtype=torch.int32, device="npu")
        context.commit(sampled, indices[:2], starts[:3], request_indices=mapping)
        for page, expected_state, expected_conv in zip(context.pages, expected_states, expected_convs):
            torch.testing.assert_close(page[1], expected_state, atol=3e-5, rtol=3e-4)
            torch.testing.assert_close(page[0], expected_conv, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["cross", "exact", "no_cross", "inplace", "zero", "full"])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("hk,hv,dk,dv", [(1, 2, 33, 47), (2, 4, 128, 128), (2, 6, 128, 128)])
def test_align_commit_restores_boundary_and_final_pages(monkeypatch, mode, dim_first, graph_mode, hk, hv, dk, dv):
    torch.manual_seed(91)
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    blocks, window, history, block_size = 8, 5, 3, 8
    channels = 2 * hk * dk + hv * dv
    computed, accepted = {
        "cross": (6, 4), "exact": (6, 2), "no_cross": (3, 2),
        "inplace": (3, 2), "zero": (6, 0), "full": (6, 5),
    }[mode]
    # Spec row 0 maps to sampler row 2; include a second request and a null row.
    mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
    counts = torch.tensor([1, 4, accepted], dtype=torch.int32, device="npu")
    computed_tokens = torch.tensor([3, 0, computed], dtype=torch.int32, device="npu")
    table_row = [1, 2, 3] if mode != "no_cross" else [2, 3, 1]
    table = torch.tensor([[4, 5, 6], [0, 0, 0], table_row], dtype=torch.int32, device="npu")
    indices = torch.tensor([1, 4, 0], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 5, 10, 10], dtype=torch.int32, device="npu")
    layers, originals, expected = [], [], []
    for layer_idx in range(2):
        # Exercise different physical channel strides in one batched commit.
        if layer_idx == 0:
            conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
        else:
            conv = paged((blocks, history + window - 1, channels), dtype=torch.bfloat16).transpose(1, 2)
        conv.copy_(torch.randn_like(conv))
        state = paged((blocks, hv, dv, dk))
        state.copy_(torch.randn_like(state) * 0.1)
        records = (paged((blocks, hv, window, dv)), paged((blocks, hk, window, dk)), paged((blocks, hv, window)))
        for record in records:
            record.fill_(float("nan"))
        q = torch.nn.functional.normalize(torch.randn(1, 10, hk, dk, device="npu"), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
        v = torch.randn(1, 10, hv, dv, device="npu", dtype=torch.bfloat16)
        g = -torch.rand(1, 10, hv, device="npu")
        beta = torch.rand(1, 10, hv, device="npu", dtype=torch.bfloat16)
        initial, old_conv = state.clone(), conv.clone()
        _, states = reference(q, k, v, g, beta, initial, [0, 5, 10, 10], [1, 4, 0])
        recovery.gdn_recoverssm_verify(q, k, v, g, beta, state, *records, starts, indices, window)
        final, final_conv = initial.clone(), old_conv.clone()
        for row, (source, n, count, destinations) in enumerate(
            [(1, computed, accepted, table_row), (4, 3, 1, [4, 5, 6])]
        ):
            if count == 0:
                continue
            boundary_len = (n // block_size + 1) * block_size - n
            targets = [(destinations[(n + count) // block_size], count)]
            if count >= boundary_len:
                targets.insert(0, (destinations[n // block_size], boundary_len))
            for target, length in targets:
                final[target] = states[row][length]
                final_conv[target, :, :history] = old_conv[source, :, length - 1 : length - 1 + history]
        layers.append(SimpleNamespace(kv_cache=(conv if dim_first else conv.transpose(1, 2), state, *records)))
        originals.append((initial, old_conv))
        expected.append((final, final_conv))
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=window, max_num_reqs=4)
    assert context.conv_dim_stride is None
    launches = []

    class RecordingKernel:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                launches.append(self.name)
                return self.kernel[grid](*args, **kwargs)

            return run

    for name in ("_commit_recoverssm_state_kernel", "_commit_recoverssm_state_grouped_kernel"):
        monkeypatch.setattr(recovery, name, RecordingKernel(name, getattr(recovery, name)))

    def launch():
        context.commit(
            counts, indices, starts, request_indices=mapping, block_table=table,
            num_computed_tokens=computed_tokens, mamba_block_size=block_size,
        )

    if graph_mode:
        launch()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()
        for page, (initial, old_conv) in zip(context.pages, originals):
            page[1].copy_(initial)
            page[0].copy_(old_conv)
        graph.replay()
    else:
        launch()
    grouped = recovery.is_950() and dk == dv == 128 and hv % 2 == 0
    selected = "_commit_recoverssm_state_grouped_kernel" if grouped else "_commit_recoverssm_state_kernel"
    assert launches and set(launches) == {selected}
    for page, (final, final_conv) in zip(context.pages, expected):
        torch.testing.assert_close(page[1], final, atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(page[0], final_conv, atol=0, rtol=0)


@pytest.mark.parametrize("ratio", [1, 2, 3])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("accepted", [1, 5, 16])
def test_reverse_commit_grouped_and_ordinary_match_forward_reference(monkeypatch, ratio, align, accepted):
    if not recovery.is_950():
        pytest.skip("Grouped state commit is selected on A5 only")
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    torch.manual_seed(117)
    blocks, hk, hv, dim, window, history = 4, 2, 2 * ratio, 128, 16, 3
    channels = 2 * hk * dim + hv * dim
    conv = paged((blocks, channels, history + window - 1), dtype=torch.bfloat16)
    conv.copy_(torch.randn_like(conv))
    state = paged((blocks, hv, dim, dim))
    state.copy_(torch.randn_like(state) * 0.1)
    corrections = paged((blocks, hv, window, dim))
    corrections.copy_(torch.randn_like(corrections) * 0.1)
    keys = paged((blocks, hk, window, dim))
    keys.copy_(torch.nn.functional.normalize(torch.randn_like(keys), dim=-1))
    gates = paged((blocks, hv, window))
    gates.copy_(-torch.rand_like(gates) * 0.1)
    layer = SimpleNamespace(kv_cache=(conv, state, corrections, keys, gates))
    context = recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=window, max_num_reqs=1)
    initial, old_conv = state.clone(), conv.clone()
    expected, expected_conv = initial.clone(), old_conv.clone()
    recovered = initial[1].double()
    for pos in range(accepted):
        key = keys[1, :, pos].double().repeat_interleave(ratio, dim=0)
        recovered = (
            recovered * gates[1, :, pos].double().exp()[:, None, None]
            + corrections[1, :, pos].double()[:, :, None] * key[:, None, :]
        )
        if align and pos + 1 == 2:
            expected[1] = recovered.float()
            expected_conv[1, :, :history] = old_conv[1, :, 1 : 1 + history]
    final_page = 2 if align and accepted >= 2 else 1
    expected[final_page] = recovered.float()
    expected_conv[final_page, :, :history] = old_conv[1, :, accepted - 1 : accepted - 1 + history]
    kwargs = {}
    if align:
        kwargs = dict(
            block_table=torch.tensor([[1, 2, 3]], dtype=torch.int32, device="npu"),
            num_computed_tokens=torch.tensor([14], dtype=torch.int32, device="npu"),
            mamba_block_size=16,
        )
    results = []
    for grouped in (False, True):
        # Both paths are valid on A5; keep the hardware-selected policy outside this comparison.
        monkeypatch.setattr(recovery, "is_950", lambda: grouped)
        state.copy_(initial)
        conv.copy_(old_conv)
        context.commit(
            torch.tensor([accepted], dtype=torch.int32, device="npu"),
            torch.tensor([1], dtype=torch.int32, device="npu"),
            torch.tensor([0, window], dtype=torch.int32, device="npu"),
            **kwargs,
        )
        torch.testing.assert_close(state, expected, atol=3e-5, rtol=3e-4)
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
        results.append(state.clone())
    torch.testing.assert_close(results[0], results[1], atol=3e-5, rtol=3e-4)


def test_empty_verification_does_not_launch_or_mutate():
    q = torch.empty(1, 0, 1, 128, dtype=torch.bfloat16, device="npu")
    v = torch.empty(1, 0, 2, 128, dtype=torch.bfloat16, device="npu")
    g = torch.empty(1, 0, 2, device="npu")
    state = torch.randn(3, 2, 128, 128, device="npu")
    before = state.clone()
    out = recovery.gdn_recoverssm_verify(
        q,
        q,
        v,
        g,
        g,
        state,
        torch.empty(3, 2, 5, 128, device="npu"),
        torch.empty(3, 1, 5, 128, device="npu"),
        torch.empty(3, 2, 5, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        torch.empty(0, dtype=torch.int32, device="npu"),
        5,
    )
    assert out.shape == v.shape
    torch.testing.assert_close(state, before, atol=0, rtol=0)


def test_nonempty_capture_rows_with_null_pages_produce_zero_outputs():
    q = torch.randn(1, 3, 1, 128, dtype=torch.bfloat16, device="npu")
    v = torch.randn(1, 3, 2, 128, dtype=torch.bfloat16, device="npu")
    g = torch.zeros(1, 3, 2, device="npu")
    state = torch.randn(3, 2, 128, 128, device="npu")
    before = state.clone()
    out = recovery.gdn_recoverssm_verify(
        q,
        q,
        v,
        g,
        g,
        state,
        torch.empty(3, 2, 5, 128, device="npu"),
        torch.empty(3, 1, 5, 128, device="npu"),
        torch.empty(3, 2, 5, device="npu"),
        torch.tensor([0, 3], dtype=torch.int32, device="npu"),
        torch.zeros(1, dtype=torch.int32, device="npu"),
        5,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    torch.testing.assert_close(state, before, atol=0, rtol=0)
