# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""AscendResizeSenderCache — P0 (前端进程) 侧自定义 resize 缓存。

参考 ShmObjectStoreSenderCache, 区别:
  - key 由 (resize 后张量, 模型信息, "resize_cache") 哈希生成
    (见 resize_cache_key.py);
  - 存储委托给 memcache (MemcacheBackend), 缓存的内容是 P1 侧 ViT 计算出的
    embedding (本进程只负责查询/占位, 填充由 P1 完成);
  - 命中时返回 marker 条目 (只含 key 与形状元数据), 不再向 P1 发送
    pixel_values, 后续 ViT 计算被跳过;
  - 未命中时先在 memcache 中 alloc 一个空对象占位, 并透传 pixel_values,
    待 P1 跑完 ViT 后回填;
  - 不实现淘汰算法, 由 memcache 组件自治。
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.multimodal.cache import (
    BaseMultiModalProcessorCache,
    MultiModalProcessorCacheInItem,
    MultiModalProcessorCacheOutItem,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)
from vllm.utils.cache import CacheInfo

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.multimodal.resize_cache_key import (
    RESIZE_CACHE_TAG,
    build_resize_cache_key,
    extract_resized_tensor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal.processing.processor import ResolvedPromptUpdate

logger = init_logger(__name__)

# marker 条目中的字段名, P1 receiver / worker patch 据此识别
# "embedding 已在 memcache 中, 跳过 ViT"
KEY_FIELD = f"{RESIZE_CACHE_TAG}_key"
NUM_TOKENS_FIELD = f"{RESIZE_CACHE_TAG}_num_tokens"
HIDDEN_DIM_FIELD = f"{RESIZE_CACHE_TAG}_hidden_dim"

_GRID_FIELDS = ("image_grid_thw", "video_grid_thw")


def _int_elem(value: int) -> MultiModalFieldElem:
    return MultiModalFieldElem(data=value, field=MultiModalBatchedField())


class AscendResizeSenderCache(BaseMultiModalProcessorCache):
    """P0 侧: resize 后按 key 查 memcache; 命中发 marker, 未命中占位并透传。"""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self._model_id: str = vllm_config.model_config.model
        self._hidden_dim = self._resolve_hidden_dim(vllm_config)
        self._merge_sq = self._resolve_merge_sq(vllm_config)
        self._elem_size = torch.empty(
            0, dtype=vllm_config.model_config.dtype
        ).element_size()
        # P0 是元数据客户端: 与 create_scheduler_client 同款用法,
        # 在 world group 建立之前初始化, 不申请存储介质
        self._backend = MemcacheBackend.create_scheduler_client(
            vllm_config.parallel_config
        )
        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    # ---- BaseMultiModalProcessorCache 接口 ----
    # 检查缓存是否存在
    def is_cached_item(self, mm_hash: str) -> bool:
        key_info_list = self._backend.batch_get_key_info([mm_hash])
        return len(key_info_list) == 0

    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        self._total += 1
        assert mm_item is not None, f"Expected processed item for {mm_hash=}"
        item, prompt_updates = mm_item

        resized = extract_resized_tensor(item)
        if resized is None:
            # 非图像/视频条目 (如 audio, prompt_embeds), 不参与本缓存
            return mm_item

        key = build_resize_cache_key(resized, self._model_id)

        if self._backend.exists([key]) == [1]:
            # 命中: memcache 中已有 P1 回填的完整 embedding 张量,
            # 返回 marker 条目, 跳过后续计算 (P1 不再跑 ViT)
            self._hits += 1
            return self._marker_item(key, item), prompt_updates

        # 未命中: 先在 memcache 中 alloc 一个空对象占位,
        # 后续 P1 跑完 ViT 后用 embedding 对它做填充;
        # 本请求仍透传 pixel_values, P1 可据此重算出同一个 key
        num_tokens = self._num_tokens(item)
        if num_tokens > 0 and self._hidden_dim > 0:
            self._backend.batch_alloc(
                [key], [num_tokens * self._hidden_dim * self._elem_size]
            )
        return mm_item

    def clear_cache(self) -> None:
        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    # ---- 内部 ----

    @staticmethod
    def _resolve_hidden_dim(vllm_config: "VllmConfig") -> int:
        # 含 deepstack 的视觉输出宽度, 与 EC connector 的 block 宽度保持一致
        from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
            _get_encoder_cache_hidden_dim,
        )

        return _get_encoder_cache_hidden_dim(vllm_config)

    @staticmethod
    def _resolve_merge_sq(vllm_config: "VllmConfig") -> int:
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        vision_config = (
            getattr(hf_config, "vision_config", None) if hf_config is not None else None
        )
        merge_size = getattr(vision_config, "spatial_merge_size", 2) or 2
        return merge_size**2

    def _num_tokens(self, item: MultiModalKwargsItem) -> int:
        for field in _GRID_FIELDS:
            if field in item:
                grid = item[field].data
                if isinstance(grid, torch.Tensor):
                    return int(grid.prod().item()) // self._merge_sq
        return 0

    def _marker_item(
        self, key: str, item: MultiModalKwargsItem
    ) -> MultiModalKwargsItem:
        """仿 ShmObjectStoreSenderCache.address_as_item: 只带 key 与形状元数据,
        供 P1 分配目标缓冲并据此从 memcache 取回 embedding。"""
        return MultiModalKwargsItem(
            {
                KEY_FIELD: MultiModalFieldElem(
                    data=key, field=MultiModalBatchedField()
                ),
                NUM_TOKENS_FIELD: _int_elem(self._num_tokens(item)),
                HIDDEN_DIM_FIELD: _int_elem(self._hidden_dim),
            }
        )
