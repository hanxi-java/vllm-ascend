# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""让 --mm-processor-cache-type custom 生效: patch vllm 的三个缓存工厂方法。

vllm 侧的 MMCacheType 已加入 "custom" (Literal 校验),
本模块负责把 "custom" 映射到 vllm-ascend 的 P0/P1 实现:
  - processor_cache_from_config       (P0 前端进程, Renderer 调用)
    → AscendResizeSenderCache
  - engine_receiver_cache_from_config (engine core 进程)
    → None (与 shm 模式一致, engine core 无需 receiver)
  - worker_receiver_cache_from_config (worker 进程, worker_base 调用)
    → AscendResizeReceiverCache

platform 与 worker 两套 patch __init__ 都会引入本模块,
保证前端进程与 worker 进程同时生效。
"""

from vllm.multimodal.registry import MultiModalRegistry

CUSTOM_TYPE = "custom"

_orig_processor_cache = MultiModalRegistry.processor_cache_from_config
_orig_engine_receiver = MultiModalRegistry.engine_receiver_cache_from_config
_orig_worker_receiver = MultiModalRegistry.worker_receiver_cache_from_config


def _processor_cache_from_config(self, vllm_config):
    if self._get_cache_type(vllm_config) == CUSTOM_TYPE:
        from vllm_ascend.multimodal.resize_sender_cache import (
            AscendResizeSenderCache,
        )

        return AscendResizeSenderCache(vllm_config)
    return _orig_processor_cache(self, vllm_config)


def _engine_receiver_cache_from_config(self, vllm_config):
    if self._get_cache_type(vllm_config) == CUSTOM_TYPE:
        return None
    return _orig_engine_receiver(self, vllm_config)


def _worker_receiver_cache_from_config(self, vllm_config, shared_worker_lock):
    if self._get_cache_type(vllm_config) == CUSTOM_TYPE:
        from vllm_ascend.multimodal.resize_receiver_cache import (
            AscendResizeReceiverCache,
        )

        # 无需 shared_worker_lock: memcache client 自身线程安全
        return AscendResizeReceiverCache(vllm_config)
    return _orig_worker_receiver(self, vllm_config, shared_worker_lock)


MultiModalRegistry.processor_cache_from_config = _processor_cache_from_config
MultiModalRegistry.engine_receiver_cache_from_config = _engine_receiver_cache_from_config
MultiModalRegistry.worker_receiver_cache_from_config = _worker_receiver_cache_from_config
