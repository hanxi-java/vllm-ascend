# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""AscendResizeReceiverCache — P1 (worker 进程) 侧自定义 resize 缓存。

参考 ShmObjectStoreReceiverCache:
  - 缓存命中 (marker 条目): 按 key 从 memcache 取出 ViT embedding,
    放入 "embedding" 字段供后续计算 (跳过 ViT);
  - 缓存未命中 (真 pixel_values): 原样透传去跑 ViT, 并对外提供
    store_embedding(), 由 worker patch 在 ViT 计算完成后将 embedding
    按地址回填到 P0 预留的 memcache 占位中;
  - key 与 P0 侧一致: 由 pixel_values 确定性重算;
  - 不实现淘汰算法, 由 memcache 组件自治。
"""

from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.logger import init_logger
from vllm.multimodal.cache import BaseMultiModalReceiverCache
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.multimodal.resize_cache_key import (
    build_resize_cache_key,
    extract_resized_tensor,
)
from vllm_ascend.multimodal.resize_sender_cache import (
    HIDDEN_DIM_FIELD,
    KEY_FIELD,
    NUM_TOKENS_FIELD,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

EMBEDDING_FIELD = "embedding"


class AscendResizeReceiverCache(BaseMultiModalReceiverCache):
    """P1 worker 侧: memcache 的读者 (取 embedding) 与填充者 (存 embedding)。"""

    # worker patch (patch_mm_resize_cache) 通过进程内单例拿到本对象;
    # TP>1 时每个 worker 进程各有一份
    _instance: ClassVar["AscendResizeReceiverCache | None"] = None

    def __init__(self, vllm_config: "VllmConfig") -> None:
        super().__init__()
        self._model_id: str = vllm_config.model_config.model
        self._dtype = vllm_config.model_config.dtype
        self._backend = MemcacheBackend(vllm_config.parallel_config)
        AscendResizeReceiverCache._instance = self

    @classmethod
    def get_instance(cls) -> "AscendResizeReceiverCache | None":
        return cls._instance

    # ---- BaseMultiModalReceiverCache 接口 ----

    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        assert mm_item is not None, f"Expected an item for {mm_hash=}"
        if KEY_FIELD in mm_item:
            # 缓存命中: 从 memcache 取出 P1 此前回填的 ViT embedding,
            # 用于后续计算 (worker patch 将其注入 encoder_cache, 跳过 ViT)
            key = mm_item[KEY_FIELD].data
            num_tokens = int(mm_item[NUM_TOKENS_FIELD].data)
            hidden_dim = int(mm_item[HIDDEN_DIM_FIELD].data)
            embedding = self.load_embedding(key, num_tokens, hidden_dim)
            return MultiModalKwargsItem(
                {
                    EMBEDDING_FIELD: MultiModalFieldElem(
                        data=embedding, field=MultiModalBatchedField()
                    ),
                    # 保留 KEY_FIELD, 供 worker patch 识别命中条目
                    KEY_FIELD: mm_item[KEY_FIELD],
                }
            )
        # 未命中: 真 pixel_values, 原样透传去跑 ViT
        return mm_item

    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        pass  # 淘汰由 memcache 自治

    def clear_cache(self) -> None:
        pass

    # ---- 对外: 供 worker patch (_execute_mm_encoder 集成) 调用 ----

    def key_for_item(self, item: MultiModalKwargsItem) -> str | None:
        """miss 路径: 从 pixel_values 确定性重算与 P0 相同的 key。"""
        resized = extract_resized_tensor(item)
        if resized is None:
            return None
        return build_resize_cache_key(resized, self._model_id)

    def load_embedding(
        self, key: str, num_tokens: int, hidden_dim: int
    ) -> torch.Tensor:
        """按地址从 memcache 读回 embedding (CPU 缓冲, 由调用方上设备)。"""
        buf = torch.empty(num_tokens, hidden_dim, dtype=self._dtype)
        self._backend.get([key], [[buf.data_ptr()]], [[buf.nbytes]])
        return buf

    def store_embedding(self, key: str, embedding: torch.Tensor) -> None:
        """将 ViT 计算出的 embedding 按地址写入 memcache, 填充 P0 预留的占位。"""
        t = embedding.contiguous()
        self._backend.put([key], [[t.data_ptr()]], [[t.nbytes]])
