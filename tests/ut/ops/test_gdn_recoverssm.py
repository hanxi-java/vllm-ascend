from pathlib import Path
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.triton import gdn_recoverssm as recovery


def make_layer(dim_first):
    conv = torch.zeros(4, 32, 7, dtype=torch.bfloat16)
    state = torch.empty(4, 2, 4, 8)
    records = (torch.empty(4, 2, 5, 4), torch.empty(4, 1, 5, 8), torch.empty(4, 2, 5))
    # Two key projections (8 channels each) plus 2*4 value channels.
    conv = conv[:, :24]
    return SimpleNamespace(kv_cache=(conv if dim_first else conv.transpose(-1, -2), state, *records))


@pytest.mark.parametrize("dim_first", [False, True])
def test_context_keeps_physical_page_strides_and_conv_orientation(monkeypatch, dim_first):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    layers = [make_layer(dim_first), make_layer(dim_first)]
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=5, max_num_reqs=4)
    assert context.conv_history_len == 3
    for index, layer in enumerate(layers):
        conv = context.pages[index][0]
        assert conv.shape == (4, 24, 7)
        assert conv.stride(0) == 32 * 7
        for offset, tensor in enumerate(context.pages[index]):
            desc = context.descriptors[index * 5 + offset]
            assert desc[0] == tensor.data_ptr()
            assert desc[1] == tensor.stride(0)
        assert context.pages[index][1] is layer.kv_cache[1]


def test_context_rejects_bf16_checkpoint_before_launch(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    layer = make_layer(True)
    layer.kv_cache = (layer.kv_cache[0], layer.kv_cache[1].bfloat16(), *layer.kv_cache[2:])
    with pytest.raises(ValueError, match="FP32 page"):
        recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)


def test_context_rejects_transposed_inner_state(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    layer = make_layer(True)
    state = torch.empty(4, 2, 8, 4).transpose(-1, -2)
    layer.kv_cache = (layer.kv_cache[0], state, *layer.kv_cache[2:])
    with pytest.raises(ValueError, match="inner dimensions"):
        recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)


def test_empty_commit_does_not_launch_and_incomplete_align_is_rejected(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    context = recovery.GDNRecoverSSMCommitContext.create([make_layer(True)], spec_query_len=5, max_num_reqs=4)
    kernel = Mock(side_effect=AssertionError("empty commit launched a kernel"))
    monkeypatch.setattr(recovery, "_commit_recoverssm_state_kernel", kernel)
    monkeypatch.setattr(recovery, "_commit_recoverssm_state_grouped_kernel", kernel)
    monkeypatch.setattr(recovery, "_prepare_commit_plan_kernel", kernel)
    monkeypatch.setattr(recovery, "_compact_conv_state_kernel", kernel)
    empty = torch.empty(0, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    context.commit(empty, empty, starts)
    with pytest.raises(ValueError, match="align metadata is incomplete"):
        context.commit(empty, empty, starts, block_table=torch.empty(0, 0))
    with pytest.raises(ValueError, match="acceptance counts"):
        context.commit(empty, torch.ones(1, dtype=torch.int32), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="sampler mapping"):
        context.commit(empty, empty, starts, request_indices=torch.ones(1, dtype=torch.int32))


@pytest.mark.parametrize("align", [False, True])
def test_commit_launches_plan_conv_state_in_order(monkeypatch, align):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(recovery, "is_950", lambda: False)
    context = recovery.GDNRecoverSSMCommitContext.create([make_layer(True)], spec_query_len=5, max_num_reqs=4)
    launches = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, grid, args, kwargs))

            return launch

    for name in ("_prepare_commit_plan_kernel", "_compact_conv_state_kernel", "_commit_recoverssm_state_kernel"):
        monkeypatch.setattr(recovery, name, Kernel(name))
    kwargs = {}
    if align:
        kwargs = dict(
            block_table=torch.tensor([[1, 2]], dtype=torch.int32),
            num_computed_tokens=torch.tensor([6], dtype=torch.int32),
            mamba_block_size=8,
        )
    context.commit(
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0, 5], dtype=torch.int32),
        **kwargs,
    )
    assert [entry[0] for entry in launches] == [
        "_prepare_commit_plan_kernel",
        "_compact_conv_state_kernel",
        "_commit_recoverssm_state_kernel",
    ]
    assert launches[0][2][6] is context.commit_lens
    assert launches[1][2][6] is context.commit_lens
    assert launches[2][3]["commit_lens_ptr"] is context.commit_lens
    assert launches[2][3]["INPUT_MODE"] == recovery.GDN_MODE
    assert launches[2][3]["NUM_KEY_HEADS"] == 1
    assert launches[1][3]["ALIGN_MODE"] is align
    assert launches[2][3]["ALIGN_MODE"] is align
    for array in (
        context.conv_state_base_addrs,
        context.conv_state_block_strides,
        context.conv_state_dim_strides,
        context.conv_state_token_strides,
    ):
        assert array.is_contiguous()
    assert context.state_layout.shape == (4, 2, 1)
    for kind in range(4):
        assert context.state_layout[kind, 0, 0] == context.pages[0][kind + 1].data_ptr()
        assert context.state_layout[kind, 1, 0] == context.pages[0][kind + 1].stride(0)


def test_align_window_and_metadata_constraints(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    context = recovery.GDNRecoverSSMCommitContext.create([make_layer(True)], spec_query_len=5, max_num_reqs=4)
    args = (torch.tensor([2]), torch.tensor([1]), torch.tensor([0, 5]))
    align = dict(block_table=torch.tensor([[1, 2]]), num_computed_tokens=torch.tensor([6]), mamba_block_size=8)
    with pytest.raises(ValueError, match="cover one speculative window"):
        context.commit(*args, **{**align, "mamba_block_size": 4})
    with pytest.raises(ValueError, match="two-dimensional"):
        context.commit(*args, **{**align, "block_table": torch.tensor([1, 2])})
    with pytest.raises(ValueError, match="integer indices"):
        context.commit(*args, **{**align, "block_table": align["block_table"].float()})
    with pytest.raises(ValueError, match="cover the sampler rows"):
        context.commit(*args, **{**align, "num_computed_tokens": torch.empty(0, dtype=torch.int32)})


def test_conv_compact_count_one_uses_shifted_history():
    """The count=1 path must shift history instead of copying in place."""
    source = """
    for pos in tl.static_range(HISTORY):
        values = tl.load(base + (count - 1 + pos) * token_stride, channels < CHANNELS, 0)
        tl.store(base + pos * token_stride, values, channels < CHANNELS)
    """
    kernel = Path("vllm_ascend/ops/triton/gdn_recoverssm.py").read_text()
    assert source.strip() in kernel
    assert "if count == 1:\n        values = tl.load(base" not in kernel
