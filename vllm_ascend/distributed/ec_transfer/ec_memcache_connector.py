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
  - L2: pHash 模糊匹配, scheduler 进程内维护字典 phash_to_mm_hash
        (pHash → (mm_hash, grid_thw)) 及配套的 band LSH 倒排索引;
        仅当 L1 无法命中时才计算本图 pHash: 汉明距离 ≤ EC_L2_MAX_HAMMING
        且 grid_thw 相同的最相似候选直接复用其 L1 条目 (近似复用;
        hamming=0 即精确匹配, 被自然包含), 仅在 served_model_name ==
        "qwenvl" 时启用, 命中即跳过 ViT。模糊命中取回后, worker 会把
        该 embedding 以本图 mm_hash 回填 memcache (后续同图请求直接
        L1 命中, 本图可用性与候选条目生命周期解耦), 并把本图 pHash
        登记进 phash_to_mm_hash (辐射后续相似图); L2 无独立的 key
        空间, embedding 按图片一份存储;
  - 两级都未命中才真正执行 ViT, 算完后以本图 mm_hash 回填 memcache
    并把 pHash 登记进 phash_to_mm_hash。

memcache 完全不感知 pHash: 读写 key 一律为 mm_hash, pHash 索引
(phash_to_mm_hash 字典 + band 倒排) 只存在于 scheduler 进程内存,
是实例级软状态, 进程重启后随 miss 重新累积 (不影响 L1 正确性)。

各钩子的分工 (scheduler 进程 / worker 进程):
  - ensure_cache_available: 调度前预计算本请求全部 mm 条目的 L1/L2 命中情况
    (此时 feature.data 可用, 可算 pHash), 使 has_cache_item 无需
    访问 request 即可给出 L1/L2 联合判定 —— 避免 patch scheduler;
  - has_cache_item: 查预计算结果 (scheduler.py:1609, 命中 → 不调度 ViT);
  - update_state_after_alloc: 命中项登记 load, 未命中项登记 save
    (scheduler.py:664-671 / :1090-1096 对两类条目都会调用);
  - build_connector_meta: 把本步 loads/saves 打包下发给 worker;
  - start_load_caches (worker): 按 loads 从 memcache 取 embedding 注入
    encoder_cache[mm_hash] (注入键永远是 mm_hash, store key 仅用于寻址);
  - save_caches (worker): ViT 算完后把 embedding 写入 memcache
    (L1: key=mm_hash; L2 模糊匹配复用候选图的 L1 条目, 无独立 L2 写)。

淘汰由 memcache 组件自治, connector 不实现任何驱逐逻辑。
"""

from dataclasses import dataclass, field
import os
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
from vllm_ascend.distributed.ec_transfer.mm_item_extract import (
    extract_image_grid,
    extract_resized_tensor,
)
from vllm_ascend.distributed.ec_transfer.phash import (
    bands,
    compute_phash,
    hamming,
)
from vllm_ascend.distributed.ec_transfer.tensor_similarity import (
    compare_tensors,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


# L2 (pHash 相似匹配) 的门控值, 与 MultiLevelEncoderCacheManager 一致
_L2_SERVED_MODEL_NAME = "qwenvl"


@dataclass
class ECMemcacheConnectorMetadata(ECConnectorMetadata):
    """Per-step scheduler → worker payload.

    loads: (mm_hash, store_key) 列表。store_key 为 L1 的 mm_hash 本身或
           L2 模糊命中候选图的 identifier; worker 取出后一律以 mm_hash
           注入 encoder_cache。store_key ≠ mm_hash (模糊命中) 时, worker
           另将该 embedding 以 mm_hash 回填 memcache, 后续同图请求直接
           L1 命中。
    saves: 本步未命中、需要 worker 在 ViT 算完后回填 L1 (key=mm_hash)
           的 mm_hash 集合。L2 无独立的 key 空间 (模糊匹配复用候选图
           的 L1 条目), 故无独立 L2 写; 模糊命中的回填在
           start_load_caches 的 load 路径完成, 不走 saves。
    """

    loads: list[tuple[str, str]] = field(default_factory=list)
    saves: set[str] = field(default_factory=set)


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
        self._similarity_enabled: bool = (
            get_served_model_name(model_config.model, model_config.served_model_name)
            == _L2_SERVED_MODEL_NAME
        )
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        self._merge_size: int = int(getattr(vision_config, "spatial_merge_size", 2) or 2)

        if role == ECConnectorRole.SCHEDULER:
            # 元数据面 client: 只做 exists 查询, 不申请存储介质
            self._backend = MemcacheBackend.create_scheduler_client(
                vllm_config.parallel_config
            )
            # 预计算的命中结果 (ensure_cache_available 填充):
            # L1 命中的 identifier 集合 (store key 即 identifier 本身)
            self._mm_hash_hits: set[str] = set()
            # L2 命中的 identifier → 候选图的 identifier,都是图片的mm_hash (worker 按它取
            # 候选图的 L1 条目, 注入键仍是本请求的 mm_hash)
            self._current_mm_hash_to_hit_mm_hash: dict[str, str] = {}
            # 累计统计
            self._full_pixel_hit_count = 0
            self._miss_count = 0
            # ── L2 模糊匹配配置与索引 (scheduler 进程内软状态) ──
            #
            # _l2_max_hamming: 模糊匹配的相似度门槛 τ, 即两个 64-bit
            #   pHash "对应位上不同的位数"的上限。0 = 仅 pHash 逐位相同
            #   才复用; 越大召回越多, 但误共享风险越高。τ 必须
            #   < band 数 (8): 距离 ≤ τ 的两个 hash, 差异位最多污染 τ
            #   个 band, 8 段中必剩至少一段完全相同 (鸽笼原理), 倒排
            #   索引才能不漏召; τ ≥ 8 时差异位可恰好一段一个, 必然
            #   漏召, 故 clamp 到 7。
            self._l2_max_hamming: int = min(
                int(os.getenv("EC_L2_MAX_HAMMING", "5")), 7)
            #
            # _band_index: LSH 倒排索引 (海选), (band 段号, 段值) →
            #   该段取该值的全部已登记 pHash。每个 pHash 切成 8 段、
            #   同时挂进 8 个桶 (有 8 次被找到的机会); 查询时拿查询
            #   hash 的 8 个 (段号, 段值) 取桶并合并去重, 即得"至少
            #   一段相同"的全部候选 —— 用固定 8 次 O(1) 查找代替与
            #   全体登记项的逐个距离计算。桶里会混入"碰巧同段"的无关
            #   hash (误候选), 由 _l2_lookup 的 hamming/grid/exists
            #   三重过滤兜底, 不误命中。
            self._band_index: dict[tuple[int, int], set[int]] = {}
            #
            # _phash_to_mm_hash: pHash 档案字典, pHash → (mm_hash, grid_thw)。
            #   它的生命周期是实例级别的，进程重启则需要重新累积。
            #   登记时机有两个, 都是 L1 未命中之后 (L1 命中不碰 pHash):
            #   ① miss 条目在 update_state_after_alloc 登记 (_l2_register);
            #   ② 模糊命中后把本图 pHash 也登记进来 (_l2_lookup) ——
            #     worker 会把候选 embedding 以本图 mm_hash 回填 memcache,
            #     本图自此可独立扮演"候选图"辐射后续相似图。
            #   同一 phash 撞 key 时先登记者赢。_band_index 只回答
            #   "有哪些候选 pHash", 本字典回答命中所需的两件事:
            #   ① 候选图 mm_hash: 候选的 embedding 以其 L1 条目
            #     (key=mm_hash) 存在 memcache 里, worker 按它取数。
            #     注意同一内容的 embedding 在相似图簇内会按图片各存
            #     一份 (模糊命中回填, 见 start_load_caches), 这是用
            #     存储换 L1 精确命中率的刻意取舍;
            #   ② grid_thw: 登记图片的尺寸标识, 做形状门控 ——
            #     encoder 输出 token 数由 grid 决定, 不同 grid 的
            #     embedding 形状不匹配, 复用会直接错, 故 grid 不同
            #     一票否决 (哪怕 pHash 完全相同)。
            self._phash_to_mm_hash: dict[int, tuple[str, tuple[int, ...]]] = {}
            #
            # _mm_hash_to_phash_cache: 步内 pHash 缓存, mm_hash → (phash, grid)。
            #   同一条目在一个调度步里要碰两次 pHash:
            #   ensure_cache_available (查询) 算一次, 未命中时
            #   update_state_after_alloc (插入索引) 还要用; 缓存避免对
            #   同一 pixel_values 算两遍 DCT。仅步内有效,
            #   build_connector_meta 时随步清空, 不跨步累积 (与上面
            #   两个长期索引的生命周期不同)。
            self._mm_hash_to_phash_cache: dict[str, tuple[int, tuple[int, ...]]] = {}
            #
            # L2 命中细分计数: hamming == 0 为精确命中 (pHash 逐位
            #   相同, 可信复用), hamming > 0 为模糊命中 (近似复用相似图
            #   的 embedding)。拆开统计用于观察模糊匹配的实际贡献;
            #   结合命中日志里的 hamming 分布, 是调整 τ 的直接依据
            #   (fuzzy 占比高但输出质量下降 → τ 定大了)。
            self._resized_exact_hit_count = 0
            self._resized_fuzzy_hit_count = 0
            # ── 调试时才使用: identifier → resized 张量注册表 ──
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
            #     - store_key: memcache 寻址用的 key。L1 命中时即本请求
            #       mm_hash 本身; L2 相似命中时为候选图的 mm_hash
            #       (worker 取回后还会以本请求 mm_hash 回填, 见
            #       start_load_caches)。
            self._mm_hashes_need_loads: list[tuple[str, str]] = []
            #
            # _mm_hashes_need_saves: 本步未命中、需要 worker 在 ViT 算完后
            #   回填 memcache 的 mm_hash 集合。worker save_caches 回调时
            #   据 mm_hash 从 encoder_cache 取刚算出的 embedding 写 L1
            #   (key=mm_hash); L2 模糊匹配复用候选图的 L1 条目, 无独立
            #   L2 写, 故此处只需登记 mm_hash。
            self._mm_hashes_need_saves: set[str] = set()
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
            self._similarity_enabled,
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
        pixel_values (算 pHash) 的标准接缝。
        """
        for feature in request.mm_features:
            current_image_mm_hash = feature.identifier
            if current_image_mm_hash in self._mm_hash_hits or current_image_mm_hash in self._current_mm_hash_to_hit_mm_hash:
                continue


            # 调试，该方法仅做过程问题定位: 登记 resized 张量并与历史条目做相似性比较.
            # self._register_and_compare_resized(
            #     request.request_id, identifier, feature.data, resized
            # )

            # L1: key = identifier (mm_hash)
            if self._backend.exists([current_image_mm_hash]) == [1]:
                self._mm_hash_hits.add(current_image_mm_hash)
                self._full_pixel_hit_count += 1
                logger.info("EC FULL-PIXEL HIT (sched): current mm_hash=%s", current_image_mm_hash)
                continue

            # L2: pHash 模糊匹配 (复用候选图的 L1 条目), 仅 qwenvl 门控内启用
            resized_pixel_tensor = extract_resized_tensor(feature.data)
            if self._similarity_enabled and resized_pixel_tensor is not None:
                logger.info("EC RESIZED SHAPE: resize_shape=%r", resized_pixel_tensor.shape)
                self._similarity_lookup(current_image_mm_hash, feature.data, resized_pixel_tensor)

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

    def _similarity_lookup(
        self,
        current_image_mm_hash: str,
        item,
        resized: torch.Tensor,
    ) -> None:
        """similarity 查找: pHash → banding 收集候选 → hamming 升序依次过
        阈值 / grid 门控 / exists 确认, 首个通过者命中 (含 hamming=0
        的精确匹配)。命中后按候选图的 identifier 复用其 L1 条目;
        候选 exists 失败只跳过, 索引软状态不会导致错命中。
        """
        grid = extract_image_grid(item)
        phash = (
            compute_phash(resized, grid, merge_size=self._merge_size)
            if grid is not None else None
        )
        if phash is None:
            return
        self._mm_hash_to_phash_cache[current_image_mm_hash] = (phash, grid)

        # 8 个 band 取桶并合并去重: "至少一段相同"的全部候选
        candidate_phash_set: set[int] = {
            h
            for band_key in bands(phash)
            for h in self._band_index.get(band_key, ())
        }

        # 候选按 hamming 距离升序排序, 首个通过阈值/grid/exists 者命中
        for candidate_phash in sorted(candidate_phash_set, key=lambda c: hamming(phash, c)):
            dist = hamming(phash, candidate_phash)
            if dist > self._l2_max_hamming:
                break
            candidate_mm_hash, candidate_grid = self._phash_to_mm_hash[candidate_phash]
            if candidate_grid != grid:
                continue  # 形状门控: token 数不同, embedding 不能复用
            if self._backend.exists([candidate_mm_hash]) != [1]:
                continue  # 候选的 L1 条目未回填完成 / 已被淘汰: 跳过
            self._current_mm_hash_to_hit_mm_hash[current_image_mm_hash] = candidate_mm_hash
            if dist == 0:
                self._resized_exact_hit_count += 1
            else:
                self._resized_fuzzy_hit_count += 1
                # 模糊命中后 worker 会把候选 embedding 以本图 identifier
                # 回填 memcache, 故把本图 pHash 也登记进字典: 后续与本图
                # 相似的请求可直接命中本图 (扩大覆盖面)。回填完成前由
                # exists 确认兜底, 不会错命中。
                self._register_phash(phash, current_image_mm_hash, grid)
            logger.info(
                "EC RESIZED-PIXEL HIT (sched): candidate_mm_hash=%s current mm_hash=%s "
                "hamming=%d phash=%016x candidate_phash=%016x",
                candidate_mm_hash, current_image_mm_hash, dist, phash, candidate_phash,
            )
            return

    def _register_phash(
        self,
        phash: int,
        current_image_mm_hash: str,
        grid: tuple[int, ...],
    ) -> None:
        """把 (pHash → (mm_hash, grid)) 登记进 phash_to_mm_hash 字典
        及 band 倒排索引; 同一 phash 撞 key 时先登记者赢。"""
        if phash in self._phash_to_mm_hash:
            return
        self._phash_to_mm_hash[phash] = (current_image_mm_hash, grid)
        for band_key in bands(phash):
            self._band_index.setdefault(band_key, set()).add(phash)

    def _similarity_register(
        self,
        current_image_mm_hash: str,
        item,
        resized: torch.Tensor,
    ) -> None:
        """未命中条目登记: 把 (pHash → (mm_hash, grid)) 插入模糊索引。

        L2 不单独存储 embedding: 本图的 embedding 由 worker 写 L1
        (key=identifier), 后续相似图命中后按本图的 identifier 复用
        该 L1 条目。插入时 embedding 尚未写入 memcache (ViT 未执行),
        后续查询有 exists 确认兜底, 指向未写入/已淘汰条目的候选只会
        被跳过。
        """
        cached_phash = self._mm_hash_to_phash_cache.get(current_image_mm_hash)
        if cached_phash is not None:
            phash, grid = cached_phash
        else:
            grid = extract_image_grid(item)
            phash = (
                compute_phash(resized, grid, merge_size=self._merge_size)
                if grid is not None else None
            )
            if phash is None:
                return
            self._mm_hash_to_phash_cache[current_image_mm_hash] = (phash, grid)

        self._register_phash(phash, current_image_mm_hash, grid)

    def has_cache_item(self, identifier: str) -> bool:
        """scheduler.py:1609 的判定: 命中则跳过 ViT 调度, 转外部加载。"""
        if identifier in self._mm_hash_hits or identifier in self._current_mm_hash_to_hit_mm_hash:
            return True
        # ensure_cache_available 未覆盖的场景 (如 running 请求的 chunk 续调度)
        # 兜底一次直接 L1 查询; L2 此时无 feature.data 可用, 放弃
        if self._backend.exists([identifier]) == [1]:
            self._mm_hash_hits.add(identifier)
            self._full_pixel_hit_count += 1
            logger.info("EC FULL-PIXEL HIT (sched, fallback): mm_hash=%s", identifier)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """命中项登记 load, 未命中项登记 save (两类条目 scheduler 都会调到)。"""
        feature = request.mm_features[index]
        current_image_mm_hash = feature.identifier

        # L1 命中: loads 二元组的取数键与注入键都是本请求 mm_hash
        if current_image_mm_hash in self._mm_hash_hits:
            self._mm_hashes_need_loads.append((current_image_mm_hash, current_image_mm_hash))
            return

        # L2 相似命中: 取数键是候选图的 mm_hash, 注入键仍是本请求 mm_hash
        if current_image_mm_hash in self._current_mm_hash_to_hit_mm_hash:
            hit_mm_hash = self._current_mm_hash_to_hit_mm_hash.get(current_image_mm_hash)
            self._mm_hashes_need_loads.append((current_image_mm_hash, hit_mm_hash))
            return

        # 未命中: ViT 将由 worker 执行, 登记 L1 回填并把 pHash 插入模糊索引
        self._miss_count += 1
        if self._similarity_enabled and feature.data is not None:
            resized = extract_resized_tensor(feature.data)
            if resized is not None:
                self._similarity_register(current_image_mm_hash, feature.data, resized)
        self._mm_hashes_need_saves.add(current_image_mm_hash)

    @property
    def hit_rate(self) -> float:
        """累计命中率: (L1 命中 + L2 精确/模糊命中) / 总判定数。"""
        hits = (self._full_pixel_hit_count
                + self._resized_exact_hit_count
                + self._resized_fuzzy_hit_count)
        total = hits + self._miss_count
        if total == 0:
            return 0.0
        return hits / total

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
                "EC meta loads: %r, EC meta saves: %r this step | "
                "full_pixel_hits=%d resized_exact_hits=%d "
                "resized_fuzzy_hits=%d misses=%d hit_rate=%.2f%%",
                len(meta.loads),
                len(meta.saves),
                meta.loads,
                meta.saves,
                self._full_pixel_hit_count,
                self._resized_exact_hit_count,
                self._resized_fuzzy_hit_count,
                self._miss_count,
                self.hit_rate * 100,
            )
        # 每步重建, 同时清空跨步状态 (累计统计字段保留)
        self._mm_hashes_need_loads = []
        self._mm_hashes_need_saves = set()
        self._mm_hash_hits.clear()
        self._current_mm_hash_to_hit_mm_hash.clear()
        self._mm_hash_to_phash_cache.clear()
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

        for current_image_mm_hash, hit_image_mm_hash in metadata.loads:
            if current_image_mm_hash in encoder_cache:
                continue
            # 取数键是 tuple 第 2 元素: L1 命中时即 mm_hash 本身;
            # L2 模糊命中时为候选图的 identifier (本请求 mm_hash 下
            # 无数据 —— 本图是命中条目, ViT 被跳过, 从未被写过,
            # 按 mm_hash 取必然取空)
            embedding = self._ec_get(hit_image_mm_hash)
            if embedding is None:
                # 调度时存在但读取时已被淘汰: 该条目将被当 miss 处理,
                # 由 encoder_cache 缺失触发后续重算 (取决于上游容错)
                logger.warning(
                    "EC LOAD miss: current_image_mm_hash=%s hit_image_mm_hash=%s (evicted?)",
                    current_image_mm_hash,
                    hit_image_mm_hash,
                )
                continue
            # 注入键永远是 mm_hash, 与 L1/L2 无关
            encoder_cache[current_image_mm_hash] = embedding
            # L2 模糊命中 (取数键 ≠ 注入键): 把候选图的 embedding 以本图
            # mm_hash 回填 memcache —— 本图后续请求直接 L1 命中, 无需再
            # 走模糊匹配; scheduler 已把本图 pHash 登记进索引, 本图也可
            # 辐射后续相似图。注意语义: 此后本图的精确 L1 命中取到的也是
            # 候选图的近似 embedding (与本图首次命中时一致, 幂等)。
            # 写仅 rank 0 执行 (同 save_caches); _ec_put 内 exists 去重。
            if hit_image_mm_hash != current_image_mm_hash and self._save_rank:
                self._ec_put(current_image_mm_hash, embedding)
            logger.info(
                "EC LOAD: hit_image_mm_hash=%s → current_image_mm_hash=%s embedding shape=%r",
                hit_image_mm_hash,
                current_image_mm_hash,
                embedding.shape,
            )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """ViT 算完后回填 L1 (key=mm_hash)。L2 模糊匹配不再单独写 L2:
        相似图经 scheduler 侧 pHash 索引直接复用本图的 L1 条目。"""
        if not self._save_rank:
            return
        if mm_hash not in encoder_cache:
            return
        self._ec_put(mm_hash, encoder_cache[mm_hash])

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
        logger.info("EC memcache get success: key=%s res shape=%r", key, res.shape)
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
