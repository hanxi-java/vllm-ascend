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
"""ECMemcacheConnector — 基于 memcache 的两级 encoder cache connector。

通过 ec_connector 框架接入 vLLM (无需修改 vllm 仓库):
    --ec-transfer-config '{
        "ec_connector": "ECMemcacheConnector",
        "ec_connector_module_path":
            "vllm_ascend.distributed.ec_transfer.ec_memcache_connector",
        "ec_role": "ec_both"
    }'

两级缓存命中规则 (逻辑平移自 MultiLevelEncoderCacheManager):
  - L1: key = mm_hash (request.mm_features[i].identifier), 命中即跳过 ViT;
  - L2: key = build_resize_cache_key(pixel_values, model_id),
        仅在 served_model_name == "qwenvl" 时启用, 命中即跳过 ViT;
  - 两级都未命中才真正执行 ViT, 算完后回填两级缓存。

各钩子的分工 (scheduler 进程 / worker 进程):
  - ensure_cache_available: 调度前预计算本请求全部 mm 条目的 L1/L2 命中情况
    (此时 feature.data 可用, 可算 resize key), 使 has_cache_item 无需
    访问 request 即可给出 L1/L2 联合判定 —— 避免 patch scheduler;
  - has_cache_item: 查预计算结果 (scheduler.py:1609, 命中 → 不调度 ViT);
  - update_state_after_alloc: 命中项登记 load, 未命中项登记 save
    (scheduler.py:664-671 / :1090-1096 对两类条目都会调用);
  - build_connector_meta: 把本步 loads/saves 打包下发给 worker;
  - start_load_caches (worker): 按 loads 从 memcache 取 embedding 注入
    encoder_cache[mm_hash] (注入键永远是 mm_hash, store key 仅用于寻址);
  - save_caches (worker): ViT 算完后把 embedding 写入 memcache
    (L1 必写, qwenvl 时 L2 也写)。

淘汰由 memcache 组件自治, connector 不实现任何驱逐逻辑。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.config.model import get_served_model_name
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)
from vllm_ascend.distributed.ec_transfer.resize_cache_key import (
    build_resize_cache_key,
    extract_image_grid,
    extract_resized_tensor,
)
from vllm_ascend.distributed.ec_transfer.tensor_similarity import (
    compare_tensors,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


# level 2 (resize cache) 的门控值, 与 MultiLevelEncoderCacheManager 一致
_L2_SERVED_MODEL_NAME = "qwenvl"


@dataclass
class ECMemcacheConnectorMetadata(ECConnectorMetadata):
    """Per-step scheduler → worker payload.

    loads: (mm_hash, store_key) 列表。store_key 为 L1 的 mm_hash 本身或
           L2 的 resize_key; worker 取出后一律以 mm_hash 注入 encoder_cache。
    saves: mm_hash → resize_key|None。worker 对每项必写 L1 (key=mm_hash),
           resize_key 非 None 时再写 L2。
    """

    loads: list[tuple[str, str]] = field(default_factory=list)
    saves: dict[str, str | None] = field(default_factory=dict)


@dataclass
class _ResizedEntry:
    """调试注册表条目: identifier 对应的 resized 张量及尺寸信息。

    grid 为 image_grid_thw (t, h, w, patch 计数), 调度阶段原始图片
    (PIL/URL) 已不保留, 无法拿到文件名, 用 grid 作图片尺寸标识。
    """

    tensor: torch.Tensor
    grid: tuple[int, ...] | None
    request_id: str


class ECMemcacheConnector(ECConnectorBase):
    """Two-level encoder cache connector backed by memcache."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        model_config = vllm_config.model_config

        self._model_id: str = model_config.model
        self._l2_enabled: bool = (
            get_served_model_name(model_config.model, model_config.served_model_name)
            == _L2_SERVED_MODEL_NAME
        )

        if role == ECConnectorRole.SCHEDULER:
            # 元数据面 client: 只做 exists 查询, 不申请存储介质
            self._backend = MemcacheBackend.create_scheduler_client(
                vllm_config.parallel_config
            )
            # 预计算的命中结果 (ensure_cache_available 填充):
            # L1 命中的 identifier 集合 (store key 即 identifier 本身)
            self._full_pixel_hits: set[str] = set()
            # L2 命中的 identifier → resize_key (worker 按 resize_key 取数)
            self._resized_pixel_hits: dict[str, str] = {}
            # 累计统计
            self._full_pixel_hit_count = 0
            self._resized_pixel_hit_count = 0
            self._miss_count = 0
            # ── 调试: identifier → resized 张量注册表 ──
            # 缓存所有首次见到的 resized 张量; 新 identifier 到来时与表内
            # 所有条目做相似性比较 (余弦相似度 + 相对误差), 结果打日志。
            # 注意: 无淘汰, 仅用于短时调试, 长期运行会持续增长内存。
            self._resized_registry: dict[str, _ResizedEntry] = {}
            # ── 本步的 loads/saves 登记簿 (build_connector_meta 打包下发后清空) ──
            #
            # _mm_hashes_need_loads: 本步缓存命中、需要 worker 从 memcache
            #   加载的条目。元素为 (mm_hash, store_key) 二元组:
            #     - mm_hash:   条目标识 (feature.identifier), worker 取出
            #       embedding 后以此注入 encoder_cache (下游
            #       _gather_mm_embeddings 只按 mm_hash 取数);
            #     - store_key: memcache 寻址用的 key。全图像素命中时即
            #       mm_hash 本身; resize 后像素命中时为 resize_key。
            self._mm_hashes_need_loads: list[tuple[str, str]] = []
            #
            # _mm_hashes_need_saves: 本步未命中、需要 worker 在 ViT 算完后
            #   回填 memcache 的条目。结构为 mm_hash → resize_key|None:
            #     - 键 mm_hash: worker save_caches 回调时据此从
            #       encoder_cache 取到刚算出的 embedding;
            #     - 值 resize_key: None 表示只写 L1 (key=mm_hash);
            #       非 None 表示 qwenvl 门控内还要再写一份 L2
            #       (key=resize_key)。scheduler 侧在此预先算好 resize_key,
            #       避免 worker 为回填重新触碰 pixel_values。
            self._mm_hashes_need_saves: dict[str, str | None] = {}
            # self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        elif role == ECConnectorRole.WORKER:
            # 数据面 client: embedding 的实际读写
            self._backend = MemcacheBackend(vllm_config.parallel_config)
            self._hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
            self._dtype = model_config.dtype
            self._elem_size = torch.empty(0, dtype=self._dtype).element_size()
            # TP 下各 rank 的 embedding 相同 (ViT 输出 all-reduce),
            # 写操作只由 rank 0 执行, 读操作各 rank 独立进行
            self._save_rank = get_world_group().rank == 0
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

        logger.info(
            "ECMemcacheConnector init: role=%s l2_enabled=%s model=%s",
            role,
            self._l2_enabled,
            self._model_id,
        )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """调度前预计算本请求全部 mm 条目的 L1/L2 命中情况。

        此钩子在 encoder 调度循环之前、携带完整 request 被调用
        (scheduler.py:838), 是唯一能同时拿到 identifier 与
        pixel_values (算 resize key) 的标准接缝。
        """
        for feature in request.mm_features:
            identifier = feature.identifier
            if identifier in self._full_pixel_hits or identifier in self._resized_pixel_hits:
                continue

            resized = extract_resized_tensor(feature.data)

            # 调试: 登记 resized 张量并与历史条目做相似性比较
            self._register_and_compare_resized(
                request.request_id, identifier, feature.data, resized
            )

            # L1: key = identifier (mm_hash)
            if self._backend.exists([identifier]) == [1]:
                self._full_pixel_hits.add(identifier)
                self._full_pixel_hit_count += 1
                logger.info("EC FULL-PIXEL HIT (sched): mm_hash=%s", identifier)
                continue

            # L2: key = resize_cache_key, 仅 qwenvl 门控内启用
            if self._l2_enabled and resized is not None:
                logger.info("EC RESIZED SHAPE: resize_shape=%r", resized.shape)
                resize_key = build_resize_cache_key(resized, self._model_id)
                if self._backend.exists([resize_key]) == [1]:
                    self._resized_pixel_hits[identifier] = resize_key
                    self._resized_pixel_hit_count += 1
                    logger.info(
                        "EC RESIZED-PIXEL HIT (sched): resize_key=%s mm_hash=%s",
                        resize_key,
                        identifier,
                    )

        # 不做延迟调度 (memcache 查询是同步的, 结果立即可用)
        return True

    def _register_and_compare_resized(
        self,
        request_id: str,
        identifier: str,
        item,
        resized: torch.Tensor | None,
    ) -> None:
        """调试: 登记 identifier 的 resized 张量, 并与注册表内所有历史
        条目两两做相似性比较 (余弦相似度 + 相对误差), 结果打日志。

        同一 identifier 只登记/比较一次; 形状不一致的条目对仅记录形状,
        不做逐元素数值比较。
        """
        if resized is None or identifier in self._resized_registry:
            return
        grid = extract_image_grid(item) if item is not None else None
        for other_id, entry in self._resized_registry.items():
            sim = compare_tensors(resized, entry.tensor)
            logger.info(
                "EC RESIZED SIMILARITY: req=%s new=%s(shape=%s grid=%s) "
                "vs cached=%s(req=%s shape=%s grid=%s) | %s",
                request_id,
                identifier,
                tuple(resized.shape),
                grid,
                other_id,
                entry.request_id,
                sim.shape_b,
                entry.grid,
                sim.format(),
            )
            if sim.top_errors:
                logger.info(
                    "EC RESIZED TOP-ERRORS: new=%s vs cached=%s (top-%d by abs_err)\n%s",
                    identifier,
                    other_id,
                    len(sim.top_errors),
                    sim.format_top_errors(),
                )
        self._resized_registry[identifier] = _ResizedEntry(
            tensor=resized.detach().cpu(),
            grid=grid,
            request_id=request_id,
        )

    def has_cache_item(self, identifier: str) -> bool:
        """scheduler.py:1609 的判定: 命中则跳过 ViT 调度, 转外部加载。"""
        if identifier in self._full_pixel_hits or identifier in self._resized_pixel_hits:
            return True
        # ensure_cache_available 未覆盖的场景 (如 running 请求的 chunk 续调度)
        # 兜底一次直接 L1 查询; L2 此时无 feature.data 可用, 放弃
        if self._backend.exists([identifier]) == [1]:
            self._full_pixel_hits.add(identifier)
            self._full_pixel_hit_count += 1
            logger.info("EC FULL-PIXEL HIT (sched, fallback): mm_hash=%s", identifier)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """命中项登记 load, 未命中项登记 save (两类条目 scheduler 都会调到)。"""
        feature = request.mm_features[index]
        identifier = feature.identifier

        if identifier in self._full_pixel_hits:
            self._full_pixel_hits.discard(identifier)
            self._mm_hashes_need_loads.append((identifier, identifier))
            return
        if identifier in self._resized_pixel_hits:
            resize_key = self._resized_pixel_hits.pop(identifier)
            self._mm_hashes_need_loads.append((identifier, resize_key))
            return

        # 未命中: ViT 将由 worker 执行, 登记回填计划
        self._miss_count += 1
        resize_key = None
        if self._l2_enabled and feature.data is not None:
            resized = extract_resized_tensor(feature.data)
            if resized is not None:
                resize_key = build_resize_cache_key(resized, self._model_id)
        self._mm_hashes_need_saves[identifier] = resize_key

    @property
    def hit_rate(self) -> float:
        """累计命中率: (L1 命中 + L2 命中) / 总判定数。"""
        total = self._full_pixel_hit_count + self._resized_pixel_hit_count + self._miss_count
        if total == 0:
            return 0.0
        return (self._full_pixel_hit_count + self._resized_pixel_hit_count) / total

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECMemcacheConnectorMetadata:
        meta = ECMemcacheConnectorMetadata(
            loads=self._mm_hashes_need_loads,
            saves=self._mm_hashes_need_saves,
        )
        if meta.loads or meta.saves:
            logger.info(
                "EC meta: %d loads, %d saves this step | "
                "full_pixel_hits=%d resized_pixel_hits=%d misses=%d hit_rate=%.2f%%",
                len(meta.loads),
                len(meta.saves),
                self._full_pixel_hit_count,
                self._resized_pixel_hit_count,
                self._miss_count,
                self.hit_rate * 100,
            )
        # 每步重建, 同时清空跨步状态 (累计统计字段保留)
        self._mm_hashes_need_loads = []
        self._mm_hashes_need_saves = {}
        self._full_pixel_hits.clear()
        self._resized_pixel_hits.clear()
        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """按 scheduler 下发的 loads 从 memcache 取 embedding, 注入 encoder_cache。"""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMemcacheConnectorMetadata)

        for mm_hash, store_key in metadata.loads:
            if mm_hash in encoder_cache:
                continue
            embedding = self._ec_get(store_key)
            if embedding is None:
                # 调度时存在但读取时已被淘汰: 该条目将被当 miss 处理,
                # 由 encoder_cache 缺失触发后续重算 (取决于上游容错)
                logger.warning(
                    "EC LOAD miss: store_key=%s mm_hash=%s (evicted?)",
                    store_key,
                    mm_hash,
                )
                continue
            # 注入键永远是 mm_hash, 与 L1/L2 无关
            encoder_cache[mm_hash] = embedding
            logger.info(
                "EC LOAD: store_key=%s → mm_hash=%s tokens=%d",
                store_key,
                mm_hash,
                embedding.shape[0],
            )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """ViT 算完后回填: L1 必写; qwenvl 且本步登记了 resize_key 时写 L2。"""
        if not self._save_rank:
            return
        if mm_hash not in encoder_cache:
            return

        embedding = encoder_cache[mm_hash]

        # L1: key = mm_hash
        self._ec_put(mm_hash, embedding)

        # L2: key = resize_key (scheduler 在 update_state_after_alloc 时算好)
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMemcacheConnectorMetadata)
        resize_key = metadata.saves.get(mm_hash)
        if resize_key is not None:
            self._ec_put(resize_key, embedding)

    # ==============================
    # Worker-side memcache helpers
    # ==============================

    def _ec_get(self, key: str) -> torch.Tensor | None:
        """按 key 从 memcache 读回 embedding (NPU 张量)。"""
        key_infos = self._backend.batch_get_key_info([key])
        if not key_infos or key_infos[0].size() == 0:
            return None
        nbytes = key_infos[0].size()
        num_tokens = nbytes // self._elem_size // self._hidden_dim
        buf = torch.empty(num_tokens, self._hidden_dim, dtype=self._dtype, device="npu")
        res = self._backend.get([key], [[buf.data_ptr()]], [[nbytes]])
        if res is None or (res and res[0] != 0):
            logger.warning("EC memcache get failed: key=%s res=%s", key, res)
            return None
        return buf

    def _ec_put(self, key: str, tensor: torch.Tensor) -> None:
        """按 key 把 embedding 写入 memcache; 已存在则跳过 (TP/重试去重)。"""
        if self._backend.exists([key]) == [1]:
            return
        t = tensor.contiguous()
        self._backend.put([key], [[t.data_ptr()]], [[t.nbytes]])
        logger.info("EC PUT: key=%s nbytes=%d", key, t.nbytes)


def _get_encoder_cache_hidden_dim(vllm_config: "VllmConfig") -> int:
    """每 token 的 encoder 输出宽度 (含 Qwen3-VL deepstack 拼接)。

    与 ec_connector/cpu/common.py 的逻辑保持一致。
    """
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None) if hf_config else None
    if vision_config is not None:
        out_hidden_size = getattr(vision_config, "out_hidden_size", None)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        if out_hidden_size is not None and deepstack_indexes:
            return out_hidden_size * (1 + len(deepstack_indexes))
    return model_config.get_inputs_embeds_size()
