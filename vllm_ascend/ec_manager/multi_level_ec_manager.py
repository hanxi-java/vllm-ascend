#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

import vllm.envs as vllm_envs
from vllm.logger import logger
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from vllm_ascend.ascend_config import get_ascend_config, get_score_encoder_cache_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
)
from vllm_ascend.multimodal.resize_cache_key import (
    build_resize_cache_key,
    extract_resized_tensor,
)
from vllm.config.model import get_served_model_name

if TYPE_CHECKING:
    from vllm.v1.request import Request


class MultiLevelEncoderCacheManager(EncoderCacheManager):
    """Encoder cache manager backed by remote memcache store.

    Differences from upstream ``EncoderCacheManager``:

    - ``check_and_update_cache`` queries memcache via ZMQ when the local
      hot-cache misses.
    - No slot capacity tracking / eviction — memcache manages its own
      storage lifecycle.
    - ``get_freed_mm_hashes`` always returns an empty list.
    """

    def __init__(self, cache_size: int):
        vllm_config = get_ascend_config().vllm_config

        backend_name = vllm_config.encoder_cache_config.get("backend", "memcache")
        backend = backend_map.get(backend_name)
        if backend is None:
            raise ValueError(f"Unsupported EC store backend: {backend_name}")
        backend_module = importlib.import_module(backend.get("path"))
        backend_class = getattr(backend_module, backend.get("name"))
        self._ec_store_client = backend_class.create_scheduler_client(
            vllm_config.parallel_config)
        model_config = vllm_config.model_config

        # 值举例： 对于命令 vllm serve /data/models/Qwen3-VL-32B --served-model-name qwenvl ... , 这个值就是  qwenvl
        self._served_model_name = get_served_model_name(
            model_config.model,
            model_config.served_model_name,
        )
        # resize 缓存 key 的模型信息成分, 必须与 P0 sender / P1 receiver
        # 两侧传入 build_resize_cache_key 的值完全一致 (同为 model 原值)
        self._model_id = model_config.model

    def check_and_update_cache(self, request: "Request", input_id: int) -> bool:
        mm_hash = request.mm_features[input_id].identifier

        # level 1: hash for the whole image. use memcache backend
        if self._ec_store_client.exists(mm_hash):
            logger.info("level 1: ec_store_client lookup memcache hit: mm_hash=%s", mm_hash)
            return True

        # level 2: image resize cache. use memcache backend.
        if self._served_model_name != "qwenvl":
            logger.info("served_model_name invalid, current is : %s", self._served_model_name)
            return False

        resize_key = self._make_resize_cache_key(request, input_id)
        if resize_key is not None and self._ec_store_client.exists(resize_key):
            logger.info("level 2: resize cache hit: resize_key=%s", resize_key)
            return True

        # level 3: 待实现.
        logger.info("EC lookup MISS (will compute): mm_hash=%s", mm_hash)
        return False

    def _make_resize_cache_key(self, request: "Request", input_id: int) -> str | None:
        """从 feature.data 中取出 resize 后的张量 (pixel_values),
        与模型信息一起生成 resize 缓存 key。

        统一委托给 vllm_ascend.multimodal.resize_cache_key.build_resize_cache_key,
        与 P0 sender / P1 receiver 使用同一把 key; image_grid_thw 不参与 hash。

        返回 None 表示无法构建 key (data 为空 / marker 条目 / 非视觉条目),
        调用方应跳过 resize 缓存查询。
        """
        data = request.mm_features[input_id].data
        if data is None:
            # lru/shm 模式命中时 P0 会置空 data 以跳过 IPC
            return None
        resized = extract_resized_tensor(data)
        if resized is None:
            # custom 模式命中时的 marker 条目 / audio 等条目没有 pixel_values
            return None
        return build_resize_cache_key(resized, self._model_id)

    def allocate(self, request: "Request", input_id: int) -> None:
        pass

    def free_encoder_input(self, request: "Request", input_id: int) -> None:
        return []

    def can_allocate(
        self,
        request: "Request",
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        return True
