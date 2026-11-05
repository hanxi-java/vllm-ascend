# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration tests can run without a vLLM or NPU installation."""

import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

helpers = runpy.run_path(str(Path(__file__).parents[3] / "vllm_ascend/models/qwen_gdn_recoverssm.py"))


def make_config():
    return NS(
        model_config=NS(
            dtype=torch.bfloat16,
            hf_text_config=NS(
                model_type="qwen3_5_moe_text",
                linear_num_key_heads=16,
                linear_num_value_heads=32,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        speculative_config=NS(method="mtp", num_speculative_tokens=4),
        use_v2_model_runner=True,
        cache_config=NS(
            use_replayssm=True,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            mamba_ssm_cache_dtype="auto",
            mamba_cache_dtype="auto",
        ),
        mamba_config=NS(enable_stochastic_rounding=False),
        parallel_config=NS(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16])
def test_target_geometry_and_model_layer_cache_agree(tp):
    config = make_config()
    config.parallel_config.tensor_parallel_size = tp
    helpers["validate_qwen_gdn_recoverssm"](config)
    original = ((8192 // tp, 7), (32 // tp, 128, 128))
    expected = (*original, (32 // tp, 5, 128), (16 // tp, 5, 128), (32 // tp, 5))

    class BaseModel:
        @classmethod
        def get_mamba_state_shape_from_config(cls, config):
            return original

        @classmethod
        def get_mamba_state_dtype_from_config(cls, config):
            return (torch.bfloat16, torch.float32)

    class Model(helpers["QwenGDNRecoverSSMModelMixin"], BaseModel):
        pass

    assert Model.get_mamba_state_shape_from_config(config) == expected
    assert helpers["append_gdn_record_shapes"](original, 16 // tp, 5) == expected
    assert Model.get_mamba_state_dtype_from_config(config) == (torch.bfloat16,) + (torch.float32,) * 4
    config.cache_config.use_replayssm = False
    assert Model.get_mamba_state_shape_from_config(config) == original
    assert Model.get_mamba_state_dtype_from_config(config) == (torch.bfloat16, torch.float32)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("speculative_config", "num_speculative_tokens", 0, "requires speculative decoding"),
        (None, "speculative_config", None, "requires speculative decoding"),
        (None, "use_v2_model_runner", False, "Runner V2"),
        ("cache_config", "mamba_ssm_cache_dtype", "bfloat16", "FP32 SSM"),
        ("cache_config", "mamba_cache_dtype", "float32", "BF16 convolution"),
        ("model_config", "dtype", torch.float16, "BF16 activations"),
        ("mamba_config", "enable_stochastic_rounding", True, "stochastic rounding"),
        ("parallel_config", "tensor_parallel_size", 3, "divide 16"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP=PCP=DCP=1"),
        ("parallel_config", "decode_context_parallel_size", 2, "PP=PCP=DCP=1"),
        (None, "kv_transfer_config", NS(kv_connector="AscendStoreConnector"), "requires Mooncake"),
    ],
)
def test_rejects_unsupported_execution_before_allocation(section, field, value, message):
    config = make_config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match=message):
        helpers["validate_qwen_gdn_recoverssm"](config)


def test_speculative_methods_share_recoverssm_validation():
    for method in ("mtp", "eagle3", "dspark"):
        config = make_config()
        config.speculative_config.method = method
        helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("method", ["mtp", "dspark", "eagle3"])
def test_pd_configuration_preserves_the_same_state_layout(connector, role, method):
    config = make_config()
    config.speculative_config.method = method
    config.parallel_config.tensor_parallel_size = 4
    config.kv_transfer_config = NS(kv_connector=connector, kv_role=role)
    helpers["validate_qwen_gdn_recoverssm"](config)

    original = ((2048, 7), (8, 128, 128))
    assert helpers["append_gdn_record_shapes"](original, 4, 5) == (
        *original,
        (8, 5, 128),
        (4, 5, 128),
        (8, 5),
    )
    assert helpers["append_gdn_record_dtypes"]((torch.bfloat16, torch.bfloat16)) == (
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.float32,
    )


@pytest.mark.parametrize("connector", ["AscendStoreConnector", "MooncakeLayerwiseConnector", "MultiConnector", None])
def test_pd_rejects_connectors_without_the_supported_state_transfer_contract(connector):
    config = make_config()
    config.kv_transfer_config = NS(kv_connector=connector, kv_role="kv_consumer")
    with pytest.raises(ValueError, match="requires MooncakeConnectorV2 or MooncakePullConnector"):
        helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("connector", ["MooncakeConnectorV2", "MooncakePullConnector"])
@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("cache_mode,prefix_caching", [("none", True), ("align", False), ("align", True)])
def test_pd_requires_none_without_prefix_caching(connector, role, cache_mode, prefix_caching):
    config = make_config()
    config.cache_config.mamba_cache_mode = cache_mode
    config.cache_config.enable_prefix_caching = prefix_caching
    config.kv_transfer_config = NS(kv_connector=connector, kv_role=role)
    with pytest.raises(ValueError, match="PD requires Mamba cache mode none and prefix caching disabled"):
        helpers["validate_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("model_type", ["qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"])
def test_target_selector_enables_qwen_gdn_architectures(model_type):
    config = make_config()
    config.model_config.hf_text_config.model_type = model_type
    assert helpers["supports_gdn_recoverssm"](config)
    assert helpers["use_qwen_gdn_recoverssm"](config)
    config.cache_config.use_replayssm = False
    assert not helpers["use_qwen_gdn_recoverssm"](config)


@pytest.mark.parametrize("model_type", ["glm5_next_text", "kimi_k3", "qwen3"])
def test_target_selector_does_not_enable_models_without_gdn_fields(model_type):
    config = make_config()
    config.model_config.hf_text_config = NS(model_type=model_type)
    assert not helpers["is_gdn_model"](config)
    assert not helpers["supports_gdn_recoverssm"](config)
    assert not helpers["use_qwen_gdn_recoverssm"](config)


def test_target_selector_accepts_derivatives_only_with_complete_gdn_fields():
    config = make_config()
    config.model_config.hf_text_config.model_type = "custom_qwen_derivative"
    assert helpers["supports_gdn_recoverssm"](config)
    assert helpers["use_qwen_gdn_recoverssm"](config)
    del config.model_config.hf_text_config.linear_conv_kernel_dim
    assert not helpers["is_gdn_model"](config)
    assert not helpers["use_qwen_gdn_recoverssm"](config)


def test_target_selector_requires_supported_geometry():
    config = make_config()
    config.model_config.hf_text_config.linear_num_key_heads = 8
    assert not helpers["supports_gdn_recoverssm"](config)
    with pytest.raises(ValueError, match="geometry"):
        helpers["validate_qwen_gdn_recoverssm"](config)


def test_align_with_prefix_caching_is_supported():
    config = make_config()
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.enable_prefix_caching = True
    helpers["validate_qwen_gdn_recoverssm"](config)
