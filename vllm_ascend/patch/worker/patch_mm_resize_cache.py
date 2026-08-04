# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""_execute_mm_encoder 集成: resize 缓存命中注入 encoder_cache, miss 回填 memcache。

仿照 gpu_model_runner 中 prompt_embeds 直通注入 encoder_cache 的先例
(gpu_model_runner.py:3015-3044):

  - 命中: feature.data 已被 AscendResizeReceiverCache 替换为含 "embedding"
    的条目 (经 _apply_mm_cache → get_and_update_features)。本 patch 把这类
    条目从 scheduled_encoder_inputs 中剔除, 其 embedding 直接
    _cache_encoder_output 注入 encoder_cache, ViT 不再执行;
  - 未命中: 条目照常走原逻辑跑 ViT, 之后按 receiver.key_for_item 重算的
    key 将 encoder output 回填 memcache (填充 P0 预留的空占位)。

仅在 AscendResizeReceiverCache 已实例化 (即 --mm-processor-cache-type
custom) 时生效, 否则完全走原逻辑。
"""

from vllm.logger import init_logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.multimodal.resize_receiver_cache import (
    EMBEDDING_FIELD,
    AscendResizeReceiverCache,
)
from vllm_ascend.multimodal.resize_sender_cache import KEY_FIELD

logger = init_logger(__name__)

_orig_execute_mm_encoder = GPUModelRunner._execute_mm_encoder


def _execute_mm_encoder_with_resize_cache(self, scheduler_output):
    receiver = AscendResizeReceiverCache.get_instance()
    if receiver is None:
        return _orig_execute_mm_encoder(self, scheduler_output)

    scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs

    # 第一遍: 拆分 命中条目 (含 KEY_FIELD, data 已被 receiver 换成 embedding)
    # 与 正常条目 (真 pixel_values, 需要跑 ViT 并回填)
    hit_embeddings = []  # (mm_hash, embedding_cpu_tensor)
    filtered_inputs = {}
    for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
        req_state = self.requests[req_id]
        kept = []
        for mm_input_id in encoder_input_ids:
            mm_feature = req_state.mm_features[mm_input_id]
            data = mm_feature.data
            if data is not None and KEY_FIELD in data:
                hit_embeddings.append(
                    (mm_feature.identifier, data[EMBEDDING_FIELD].data)
                )
            else:
                kept.append(mm_input_id)
        if kept:
            filtered_inputs[req_id] = kept

    # 命中的 embedding 直接注入 encoder_cache, 跳过 ViT
    # (与 prompt_embeds 直通路径相同的注入方式)
    for mm_hash, embedding in hit_embeddings:
        self._cache_encoder_output(
            mm_hash,
            embedding.to(self.device),
            scheduler_output.ec_manager_metadata,
            scheduler_output.free_encoder_mm_hashes,
        )

    # 第二遍: 在正常条目上预计算回填用的 key
    # (与 _batch_mm_inputs_from_scheduler 相同的迭代顺序与过滤条件,
    #  再排除 orig 内部会过滤掉的 prompt_embeds, 保证与 encoder_outputs 对齐)
    pending_keys = []  # 与 orig 返回的 encoder_outputs 一一对应; None 表示不回填
    for req_id, encoder_input_ids in filtered_inputs.items():
        req_state = self.requests[req_id]
        for mm_input_id in encoder_input_ids:
            mm_feature = req_state.mm_features[mm_input_id]
            data = mm_feature.data
            if data is None or mm_feature.modality == "prompt_embeds":
                continue
            pending_keys.append(receiver.key_for_item(data))

    # 用过滤后的 scheduled_encoder_inputs 跑原逻辑 (单线程 worker, 临时替换)
    scheduler_output.scheduled_encoder_inputs = filtered_inputs
    try:
        encoder_outputs = _orig_execute_mm_encoder(self, scheduler_output)
    finally:
        scheduler_output.scheduled_encoder_inputs = scheduled_encoder_inputs

    # miss 回填: 把 ViT 算出的 embedding 写入 P0 预留的 memcache 占位
    for key, output in zip(pending_keys, encoder_outputs):
        if key is not None:
            receiver.store_embedding(key, output)

    return encoder_outputs


GPUModelRunner._execute_mm_encoder = _execute_mm_encoder_with_resize_cache
