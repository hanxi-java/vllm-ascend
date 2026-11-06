# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen GDN RecoverSSM configuration and paged record layout.

Keep this module free of worker/operator imports: platform validation runs
before NPU workers and their custom operators are initialized.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_GDN_MODEL_TYPES = frozenset(
    {
        "qwen3_5_moe_text",
        "qwen3_5_text",
        "qwen3_next",
    }
)


def is_gdn_model(config: "VllmConfig") -> bool:
    """Return whether the model exposes the Qwen GDN geometry/configuration.

    Keep this capability check local to the GDN path.  KDA's selector and
    validation remain untouched, so adding another GDN model type cannot
    change the KDA RecoverSSM path.
    """
    model = config.model_config
    hf = getattr(model, "hf_text_config", None) if model is not None else None
    if hf is None:
        return False
    model_type = getattr(hf, "model_type", None)
    if model_type in _GDN_MODEL_TYPES:
        return True
    # Some Qwen derivatives use a custom model_type but retain the GDN
    # configuration fields.  Do not classify arbitrary models without the
    # complete set of GDN head fields.
    return all(
        hasattr(hf, name)
        for name in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
    )

def use_qwen_gdn_recoverssm(config: "VllmConfig") -> bool:
    return config.cache_config.use_replayssm and is_gdn_model(config)

def validate_qwen_gdn_recoverssm(config: "VllmConfig") -> None:
    spec = config.speculative_config
    if spec is None or spec.num_speculative_tokens < 1:
        raise ValueError("Qwen GDN RecoverSSM requires speculative decoding with at least one speculative token")
    if not config.use_v2_model_runner:
        raise ValueError("Qwen GDN RecoverSSM requires Model Runner V2")
    cache = config.cache_config
    if config.mamba_config.enable_stochastic_rounding:
        raise ValueError("Qwen GDN RecoverSSM does not support stochastic rounding")
    if config.model_config.dtype != torch.bfloat16 or cache.mamba_ssm_cache_dtype not in ("auto", "float32"):
        raise ValueError("Qwen GDN RecoverSSM requires BF16 activations and FP32 SSM state")
    if cache.mamba_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Qwen GDN RecoverSSM requires BF16 convolution state")
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("Qwen GDN RecoverSSM requires PP=PCP=DCP=1")
    transfer = config.kv_transfer_config
    if transfer is not None:
        if transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector"):
            raise ValueError("Qwen GDN RecoverSSM KV transfer requires MooncakeConnectorV2 or MooncakePullConnector")
        if cache.mamba_cache_mode != "none" or cache.enable_prefix_caching:
            raise ValueError("Qwen GDN RecoverSSM PD requires Mamba cache mode none and prefix caching disabled")
    else:
        expected_cache_mode = "align" if cache.enable_prefix_caching else "none"
        if cache.mamba_cache_mode != expected_cache_mode:
            raise ValueError(
                f"Qwen GDN RecoverSSM requires Mamba cache mode {expected_cache_mode} "
                "with this prefix caching setting"
            )
    if 16 % parallel.tensor_parallel_size:
        raise ValueError("Qwen GDN RecoverSSM TP size must divide 16 key heads")

def append_gdn_record_shapes(shapes, num_key_heads: int, spec_query_len: int):
    """Append correction, effective key and log-decay records to conv/state."""
    _, (num_value_heads, value_dim, key_dim) = shapes
    return (
        *shapes,
        (num_value_heads, spec_query_len, value_dim),
        (num_key_heads, spec_query_len, key_dim),
        (num_value_heads, spec_query_len),
    )


def append_gdn_record_dtypes(dtypes):
    # Force the checkpoint to FP32 even when an unresolved "auto" reached us.
    return (dtypes[0], torch.float32, torch.float32, torch.float32, torch.float32)


def gdn_recoverssm_state_shapes(shapes, config: "VllmConfig"):
    """Append RecoverSSM record pages when the feature is active."""
    if not use_qwen_gdn_recoverssm(config):
        return shapes
    return append_gdn_record_shapes(
        shapes,
        config.model_config.hf_text_config.linear_num_key_heads // config.parallel_config.tensor_parallel_size,
        config.speculative_config.num_speculative_tokens + 1,
    )


def gdn_recoverssm_state_dtypes(dtypes, config: "VllmConfig"):
    return append_gdn_record_dtypes(dtypes) if use_qwen_gdn_recoverssm(config) else dtypes


class QwenGDNRecoverSSMModelMixin:
    @classmethod
    def get_mamba_state_shape_from_config(cls, config):
        return gdn_recoverssm_state_shapes(super().get_mamba_state_shape_from_config(config), config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, config):
        return gdn_recoverssm_state_dtypes(super().get_mamba_state_dtype_from_config(config), config)
