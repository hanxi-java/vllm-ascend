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
  - L2: 模糊匹配, matcher 由 EC_L2_MATCHER 的值决定，有以下选择 ("ssim" 默认 /
        "phash" / "phash_ssim"):
        ① phash: scheduler 进程内维护字典 phash_to_mm_hash
           (pHash → (mm_hash, grid_thw)) 及配套的 band LSH 倒排索引;
           仅当 L1 无法命中时才计算本图 pHash: 汉明距离 ≤ EC_L2_MAX_HAMMING
           且 grid_thw 相同的最相似候选直接复用其 L1 条目 (近似复用;
           hamming=0 即精确匹配, 被自然包含);
        ② ssim(仅测试用): 历史图片的 resized 张量以 "'resized_' 前缀 + mm_hash" key
           写入 memcache (淘汰随 memcache 自治), scheduler 本地只留
           _mm_hash_to_resized_meta 字典 (mm_hash → (shape, dtype))
           作枚举与读回重建用; 新请求 L1 未命中后与表内形状相同的
           条目逐一读回并计算 patch 灰度平面 SSIM (候选灰度平面用
           查询 grid 现算, 形状对不上的条目被自然跳过; 逐次计时打日志),
           得分 ≥ EC_SSIM_MIN_SCORE(默认 0.99) 的候选按得分降序取首个
           exists 确认者复用其 L1 条目; data_range 由
           EC_SSIM_DATA_RANGE 控制 (默认 3.7, 归一化 pixel_values
           的单通道理论值域 1/std);
        ③ phash_ssim: 两级串联, 兼顾 pHash 索引的召回效率与 SSIM 的
           精度 —— 先按 ① 的 band LSH 倒排索引海选 hamming ≤
           EC_L2_MAX_HAMMING 且 grid_thw 相同的候选 (固定 8 次 O(1)
           查找代替全量两两比较), 再仅对这些候选按 ② 的 SSIM 精排,
           得分 ≥ EC_SSIM_MIN_SCORE 且 exists 确认者命中; pHash 索引
           与 SSIM 注册表同时维护 (登记时机相同);
        三种 matcher 均由环境变量 EC_L2_ENABLED 开启 ("1"/"true"/
        "yes"/"on", 默认关闭),
        命中即跳过 ViT。模糊命中取回后, worker 会把该 embedding 以本图
        mm_hash 回填 memcache (后续同图请求直接 L1 命中, 本图可用性与
        候选条目生命周期解耦), 并把本图登记进对应索引 (辐射后续相似图);
        L2 无独立的 key 空间, embedding 按图片一份存储;
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
import threading
import time
from typing import TYPE_CHECKING

import torch
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
from vllm_ascend.distributed.ec_transfer.ssim import (
    DEFAULT_DATA_RANGE,
    patch_gray_plane,
    ssim_score,
)
from vllm_ascend.distributed.ec_transfer.tensor_similarity import (
    compare_tensors,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


# scheduler 侧写入 memcache 的 resized 张量 key 前缀 (与 L1 embedding
# 的 mm_hash key 空间隔离, 避免撞 key)
_RESIZED_KEY_PREFIX = "resized_"


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
        # L2 模糊匹配总开关: EC_L2_ENABLED 环境变量, 默认关闭
        self._similarity_enabled: bool = (
            os.getenv("EC_L2_ENABLED", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        self._merge_size: int = int(getattr(vision_config, "spatial_merge_size", 2) or 2)

        if role == ECConnectorRole.SCHEDULER:
            # 元数据面 client: exists 查询 + resized 张量读写, 不申请存储介质
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
            # ── L2 matcher 选择与 SSIM 配置 ──
            #
            # _l2_matcher: L2 模糊匹配方式, 三选一:
            #   "phash":      仅 pHash (band LSH 海选 + hamming 门控);
            #   "ssim" (默认, 测试用): 仅 SSIM 全量两两比较;
            #   "phash_ssim": pHash 海选候选 + SSIM 精排确认 (串联)。
            self._l2_matcher: str = os.getenv("EC_L2_MATCHER", "ssim").strip().lower()
            if self._l2_matcher not in ("ssim", "phash", "phash_ssim"):
                logger.warning("Unknown EC_L2_MATCHER=%s, fallback to ssim",
                               self._l2_matcher)
                self._l2_matcher = "ssim"
            #
            # _ssim_threshold: SSIM 相似判定阈值 τ。0.99 是经验起点,
            #   需按正例 (重压缩/resize 往返) 与负例 (同版式不同内容)
            #   的分数分布标定; 假阳性会把候选图 embedding 静默注入给
            #   本图, τ 应偏向"宁可 miss 不可错命中"。
            self._ssim_threshold: float = float(os.getenv("EC_SSIM_MIN_SCORE", "0.99"))
            #
            # _ssim_data_range: SSIM 稳定常数的值域基准 L。归一化
            #   pixel_values 单通道理论值域 = 1/std ≈ 3.7 (CLIP std);
            #   必须全局固定, 不能逐对自适应 (否则分数间不可比)。
            self._ssim_data_range: float = float(
                os.getenv("EC_SSIM_DATA_RANGE", str(DEFAULT_DATA_RANGE)))
            #
            # _mm_hash_to_resized_meta: 历史图片 resized 张量的本地元信息
            #   (测试用), mm_hash → (shape, dtype)。resized 张量本身以
            #   "resized_" 前缀 key 写入 memcache (见 _ssim_register),
            #   淘汰随 memcache 自治; 本字典只用于枚举已登记条目和读回
            #   时重建张量, 是实例级软状态, 进程重启后随 miss 重新累积
            #   (memcache 里残留的 resized 条目无本地元信息, 不会被
            #   枚举到, 仅多占存储)。登记时机与 pHash 索引相同: miss
            #   条目在 update_state_after_alloc 登记, 模糊命中后把本图
            #   也登记进来 (worker 回填后可辐射后续相似图)。比较时先按
            #   shape 预筛, 再从 memcache 读回候选张量, 用查询 grid
            #   现算 patch 灰度平面做 SSIM (形状对不上的被自然跳过)。
            #   无淘汰, 仅用于短时测试, 长期运行会持续增长内存。
            self._mm_hash_to_resized_meta: dict[
                str, tuple[tuple[int, ...], torch.dtype]] = {}
            #
            # SSIM 单次比较耗时统计 (累计, 供观测调度路径开销)。
            self._ssim_cmp_count = 0
            self._ssim_cmp_total_ms = 0.0
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

            # L2: 模糊匹配 (复用候选图的 L1 条目), 仅 EC_L2_ENABLED 开启时启用
            resized_pixel_tensor = extract_resized_tensor(feature.data)
            if self._similarity_enabled and resized_pixel_tensor is not None:
                logger.info("EC RESIZED SHAPE: resize_shape=%r", resized_pixel_tensor.shape)
                self._l2_lookup(current_image_mm_hash, feature.data,
                                resized_pixel_tensor)

        # 不做延迟调度 (memcache 查询是同步的, 结果立即可用)
        return True

    def _l2_lookup(
        self,
        current_image_mm_hash: str,
        item,
        resized: torch.Tensor,
    ) -> None:
        """按 EC_L2_MATCHER 分派 L2 模糊查找。"""
        if self._l2_matcher == "ssim":
            self._ssim_lookup(current_image_mm_hash, item, resized)
        elif self._l2_matcher == "phash":
            self._similarity_lookup(current_image_mm_hash, item, resized)
        else:  # phash_ssim: pHash 海选 + SSIM 精排
            self._phash_ssim_lookup(current_image_mm_hash, item, resized)

    def _l2_register(
        self,
        current_image_mm_hash: str,
        item,
        resized: torch.Tensor,
    ) -> None:
        """按 EC_L2_MATCHER 分派未命中条目的索引登记 (phash_ssim
        模式下 pHash 索引与 SSIM 注册表都登记)。"""
        if self._l2_matcher == "ssim":
            self._ssim_register(current_image_mm_hash, resized)
        elif self._l2_matcher == "phash":
            self._similarity_register(current_image_mm_hash, item, resized)
        else:  # phash_ssim
            self._similarity_register(current_image_mm_hash, item, resized)
            self._ssim_register(current_image_mm_hash, resized)

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

    def _compute_phash(
        self,
        current_image_mm_hash: str,
        item,
        resized: torch.Tensor,
    ) -> tuple[int, tuple[int, ...]] | None:
        """计算本图 pHash (带步内缓存): 同一条目一个调度步内查询
        (ensure_cache_available) 与登记 (update_state_after_alloc)
        各用一次, 缓存避免对同一 pixel_values 算两遍 DCT。
        grid 提取失败或 pHash 不可算时返回 None。
        """
        cached = self._mm_hash_to_phash_cache.get(current_image_mm_hash)
        if cached is not None:
            return cached
        grid = extract_image_grid(item)
        phash = (
            compute_phash(resized, grid, merge_size=self._merge_size)
            if grid is not None else None
        )
        if phash is None:
            return None
        self._mm_hash_to_phash_cache[current_image_mm_hash] = (phash, grid)
        return phash, grid

    def _phash_candidates(
        self,
        phash: int,
        grid: tuple[int, ...],
        exclude: str | None = None,
    ) -> list[tuple[int, int, str]]:
        """band 倒排海选: 8 个 band 取桶并合并去重得"至少一段相同"的
        全部候选, 按 hamming 升序过阈值/grid 门控, 返回
        [(hamming, candidate_phash, candidate_mm_hash)]。

        桶里会混入"碰巧同段"的无关 hash (误候选), 由后续 exists/SSIM
        过滤兜底, 不误命中。
        """
        candidate_phash_set: set[int] = {
            h
            for band_key in bands(phash)
            for h in self._band_index.get(band_key, ())
        }
        candidates: list[tuple[int, int, str]] = []
        for candidate_phash in sorted(candidate_phash_set, key=lambda c: hamming(phash, c)):
            dist = hamming(phash, candidate_phash)
            if dist > self._l2_max_hamming:
                break
            candidate_mm_hash, candidate_grid = self._phash_to_mm_hash[candidate_phash]
            if candidate_grid != grid:
                continue  # 形状门控: token 数不同, embedding 不能复用
            if candidate_mm_hash == exclude:
                continue
            candidates.append((dist, candidate_phash, candidate_mm_hash))
        return candidates

    def _record_l2_hit(
        self,
        current_image_mm_hash: str,
        candidate_mm_hash: str,
        exact: bool = False,
    ) -> None:
        """登记 L2 命中: current → candidate 注入映射 + 精确/模糊计数。"""
        self._current_mm_hash_to_hit_mm_hash[current_image_mm_hash] = candidate_mm_hash
        if exact:
            self._resized_exact_hit_count += 1
        else:
            self._resized_fuzzy_hit_count += 1

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
        result = self._compute_phash(current_image_mm_hash, item, resized)
        if result is None:
            return
        phash, grid = result

        # 候选按 hamming 距离升序, 首个通过 exists 确认者命中
        for dist, candidate_phash, candidate_mm_hash in self._phash_candidates(
                phash, grid, exclude=current_image_mm_hash):
            if self._backend.exists([candidate_mm_hash]) != [1]:
                continue  # 候选的 L1 条目未回填完成 / 已被淘汰: 跳过
            self._record_l2_hit(current_image_mm_hash, candidate_mm_hash,
                                exact=(dist == 0))
            if dist > 0:
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

    def _ssim_score_candidates(
        self,
        current_image_mm_hash: str,
        candidate_mm_hashes: list[str],
        query_gray: torch.Tensor,
        grid: tuple[int, ...],
        log_tag: str,
    ) -> list[tuple[float, str]]:
        """对候选逐一 SSIM 打分: memcache 读回 resized 张量 → 用查询
        grid 现算灰度平面 → ssim_score (逐次 perf_counter 计时打日志);
        返回得分 ≥ 阈值的 [(score, candidate_mm_hash)] (未排序)。
        读回失败 / 形状对不上的候选跳过。
        """
        over_threshold: list[tuple[float, str]] = []
        for candidate_mm_hash in candidate_mm_hashes:
            t0 = time.perf_counter()
            candidate_resized = self._resized_get(candidate_mm_hash)
            if candidate_resized is None:
                continue  # resized 条目未回填完成 / 已被淘汰: 跳过
            candidate_gray = patch_gray_plane(candidate_resized, grid,
                                              merge_size=self._merge_size)
            if candidate_gray is None:
                continue
            score = ssim_score(query_gray, candidate_gray,
                               data_range=self._ssim_data_range)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            if score is None:
                continue
            self._ssim_cmp_count += 1
            self._ssim_cmp_total_ms += elapsed_ms
            logger.info(
                "%s CMP: current=%s vs candidate=%s ssim=%.6f elapsed=%.3fms",
                log_tag, current_image_mm_hash, candidate_mm_hash, score,
                elapsed_ms,
            )
            if score >= self._ssim_threshold:
                over_threshold.append((score, candidate_mm_hash))
        return over_threshold

    def _first_existing_candidate(
        self,
        over_threshold: list[tuple[float, str]],
    ) -> tuple[float, str] | None:
        """得分降序取首个 L1 条目 exists 确认的候选; 无则 None
        (候选条目未回填完成 / 已被淘汰时取次优)。"""
        for score, candidate_mm_hash in sorted(over_threshold, key=lambda s: -s[0]):
            if self._backend.exists([candidate_mm_hash]) == [1]:
                return score, candidate_mm_hash
        return None

    def _ssim_lookup(
        self,
        current_image_mm_hash: str,
        item,
        resized: torch.Tensor,
    ) -> None:
        """SSIM 查找: 与注册表内形状相同的历史条目逐一计算 SSIM (候选
        张量从 memcache 读回, 灰度平面用查询 grid 现算), 得分 ≥ 阈值者
        按降序取首个 exists 确认的候选命中。测试用: 全量两两比较,
        无索引加速。
        """
        grid = extract_image_grid(item)
        if grid is None:
            return
        query_gray = patch_gray_plane(resized, grid, merge_size=self._merge_size)
        if query_gray is None:
            return

        t_lookup = time.perf_counter()
        # 形状门控: token 数不同, embedding 不能复用
        query_shape = tuple(resized.shape)
        candidates = [
            mm_hash
            for mm_hash, (shape, _) in self._mm_hash_to_resized_meta.items()
            if mm_hash != current_image_mm_hash and shape == query_shape
        ]
        over_threshold = self._ssim_score_candidates(
            current_image_mm_hash, candidates, query_gray, grid, "EC SSIM")
        logger.info(
            "EC SSIM LOOKUP: current=%s registry=%d candidates=%d "
            "over_threshold=%d total=%.3fms",
            current_image_mm_hash, len(self._mm_hash_to_resized_meta),
            len(candidates), len(over_threshold),
            (time.perf_counter() - t_lookup) * 1e3,
        )

        hit = self._first_existing_candidate(over_threshold)
        if hit is None:
            return
        score, candidate_mm_hash = hit
        self._record_l2_hit(current_image_mm_hash, candidate_mm_hash)
        # 模糊命中后 worker 会把候选 embedding 以本图 mm_hash 回填
        # memcache, 故把本图也登记进注册表: 后续与本图相似的请求
        # 可直接命中本图 (扩大覆盖面)。回填完成前由 exists 确认兜底。
        self._ssim_register(current_image_mm_hash, resized)
        logger.info(
            "EC SSIM HIT (sched): candidate_mm_hash=%s current mm_hash=%s "
            "ssim=%.6f threshold=%.4f",
            candidate_mm_hash, current_image_mm_hash, score,
            self._ssim_threshold,
        )

    def _phash_ssim_lookup(
        self,
        current_image_mm_hash: str,
        item,
        resized: torch.Tensor,
    ) -> None:
        """pHash+SSIM 串联查找: 先用 band LSH 倒排索引海选 hamming ≤ τ
        且 grid 相同的候选 (固定 8 次 O(1) 查找, 避免全量两两比较), 再
        仅对这些候选逐一计算 SSIM 精排; 得分 ≥ 阈值且首个通过 exists
        确认者命中。

        与纯 phash 路径的差异是多一道 SSIM 精排: pHash 海选召回的"碰巧
        相似"候选 (hamming 低但内容不同) 会被 SSIM 滤掉, 降低误共享风险;
        与纯 ssim 路径的差异是候选集由索引给出, SSIM 比较次数从 O(注册表
        全量) 降为 O(候选数)。
        """
        result = self._compute_phash(current_image_mm_hash, item, resized)
        if result is None:
            return
        phash, grid = result

        candidates = [
            candidate_mm_hash
            for _, _, candidate_mm_hash in self._phash_candidates(
                phash, grid, exclude=current_image_mm_hash)
        ]
        if not candidates:
            return

        # SSIM 精排: 仅对海选候选计算, 得分 ≥ 阈值者按降序取首个
        # exists 确认者命中
        query_gray = patch_gray_plane(resized, grid, merge_size=self._merge_size)
        if query_gray is None:
            return
        over_threshold = self._ssim_score_candidates(
            current_image_mm_hash, candidates, query_gray, grid,
            "EC PHASH+SSIM")
        hit = self._first_existing_candidate(over_threshold)
        if hit is None:
            return
        score, candidate_mm_hash = hit
        self._record_l2_hit(current_image_mm_hash, candidate_mm_hash)
        # 命中后 worker 会把候选 embedding 以本图 mm_hash 回填
        # memcache, 故把本图同时登记进 pHash 索引与 SSIM 注册表:
        # 后续与本图相似的请求可直接命中本图 (扩大覆盖面)。
        self._register_phash(phash, current_image_mm_hash, grid)
        self._ssim_register(current_image_mm_hash, resized)
        logger.info(
            "EC PHASH+SSIM HIT (sched): candidate_mm_hash=%s "
            "current mm_hash=%s ssim=%.6f threshold=%.4f",
            candidate_mm_hash, current_image_mm_hash, score,
            self._ssim_threshold,
        )

    def _ssim_register(
        self,
        current_image_mm_hash: str,
        resized: torch.Tensor,
    ) -> None:
        """把历史图片的 resized 张量异步写入 memcache (key 加
        "resized_" 前缀), 本地同步登记 (shape, dtype) 元信息供枚举与
        读回重建 (测试用, 无淘汰)。

        登记拆成两步: 元信息登记在主流程同步完成 (开销可忽略, 且使
        _mm_hash_to_resized_meta 保持主线程单写者, 无需加锁; 先登记
        也防止同图重复起线程); memcache 写入是网络操作, offload 到
        daemon 线程异步执行, 不阻塞调度主流程。插入时 embedding 可能
        尚未写入 memcache (ViT 未执行), 写入线程也可能尚未完成, 后续
        查询有 exists / _resized_get 读回确认兜底, 指向未写入/已淘汰
        条目的候选只会被跳过。
        """
        if current_image_mm_hash in self._mm_hash_to_resized_meta:
            return
        self._mm_hash_to_resized_meta[current_image_mm_hash] = (
            tuple(resized.shape), resized.dtype)
        threading.Thread(
            target=self._ssim_register_async,
            args=(current_image_mm_hash, resized),
            daemon=True,  # 不阻塞进程退出; 未完成的写入按丢失处理
        ).start()

    def _ssim_register_async(
        self,
        current_image_mm_hash: str,
        resized: torch.Tensor,
    ) -> None:
        """_ssim_register 的后台线程体: 拷贝 + memcache 写入。
        异常只打日志不回传 —— 写入失败的候选后续 _resized_get
        读不到即自然跳过, 不影响主流程正确性。
        """
        try:
            t = resized.detach().cpu().contiguous()
            self._resized_put(current_image_mm_hash, t)
        except Exception:
            logger.exception(
                "EC RESIZED async register failed: mm_hash=%s",
                current_image_mm_hash,
            )

    # ==============================
    # Scheduler-side memcache helpers (resized 张量)
    # ==============================

    def _resized_put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        """把 resized 张量以 "resized_" 前缀 key 写入 memcache;
        已存在则跳过 (去重)。"""
        key = _RESIZED_KEY_PREFIX + mm_hash
        if self._backend.exists([key]) == [1]:
            return
        t = tensor.contiguous()
        # CPU 内存的 put 是同步拷贝, 无 _ec_put 的计算流/SDMA 队列同步
        # 问题; 只需保证 t 存活到 put 返回
        self._backend.put([key], [[t.data_ptr()]], [[t.nbytes]])
        logger.info("EC RESIZED PUT: key=%s nbytes=%d shape=%r",
                    key, t.nbytes, tuple(t.shape))

    def _resized_get(self, mm_hash: str) -> torch.Tensor | None:
        """按 mm_hash 从 memcache 读回 resized 张量 (CPU); 无本地元信息 /
        条目缺失 / 尺寸不符 / 读失败时返回 None。"""
        meta = self._mm_hash_to_resized_meta.get(mm_hash)
        if meta is None:
            return None
        shape, dtype = meta
        key = _RESIZED_KEY_PREFIX + mm_hash
        key_infos = self._backend.batch_get_key_info([key])
        if not key_infos or key_infos[0].size() == 0:
            return None
        nbytes = key_infos[0].size()
        buf = torch.empty(shape, dtype=dtype)
        if buf.nbytes != nbytes:
            logger.warning(
                "EC RESIZED GET size mismatch: key=%s meta=%r/%s nbytes=%d "
                "store_nbytes=%d", key, shape, dtype, buf.nbytes, nbytes)
            return None
        res = self._backend.get([key], [[buf.data_ptr()]], [[nbytes]])
        if res is None or (res and res[0] != 0):
            logger.warning("EC RESIZED GET failed: key=%s res=%s", key, res)
            return None
        # 目标是 CPU 内存, get 返回即拷贝完成, 无需 _ec_get 的
        # torch.npu.synchronize (那是等 SDMA 直写 NPU buffer)
        return buf

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
        result = self._compute_phash(current_image_mm_hash, item, resized)
        if result is None:
            return
        phash, grid = result
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

        # 未命中: ViT 将由 worker 执行, 登记 L1 回填并把本图插入模糊索引
        self._miss_count += 1
        if self._similarity_enabled and feature.data is not None:
            resized = extract_resized_tensor(feature.data)
            if resized is not None:
                self._l2_register(current_image_mm_hash, feature.data, resized)
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
            ssim_avg_ms = (self._ssim_cmp_total_ms / self._ssim_cmp_count
                           if self._ssim_cmp_count else 0.0)
            logger.info(
                "EC meta: %d loads, %d saves this step | "
                "EC meta loads: %r, EC meta saves: %r this step | "
                "full_pixel_hits=%d resized_exact_hits=%d "
                "resized_fuzzy_hits=%d misses=%d hit_rate=%.2f%% | "
                "l2_matcher=%s ssim_cmps=%d ssim_avg_ms=%.3f",
                len(meta.loads),
                len(meta.saves),
                meta.loads,
                meta.saves,
                self._full_pixel_hit_count,
                self._resized_exact_hit_count,
                self._resized_fuzzy_hit_count,
                self._miss_count,
                self.hit_rate * 100,
                self._l2_matcher,
                self._ssim_cmp_count,
                ssim_avg_ms,
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
            # 注入键永远是 mm_hash
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
        # get 是 SDMA 引擎直写 buf, 与后续 forward 读 buf 的计算流分属
        # 不同硬件队列; 注入 encoder_cache 前同步一次, 保证计算流读到
        # 完整的拷贝结果, 否则模型拿到的是未写完的 buffer (垃圾输出)。
        torch.npu.synchronize()
        logger.info("EC memcache get success: key=%s buf shape=%r", key, buf.shape)
        return buf

    def _ec_put(self, key: str, tensor: torch.Tensor) -> None:
        """按 key 把 embedding 写入 memcache; 已存在则跳过 (TP/重试去重)。"""
        if self._backend.exists([key]) == [1]:
            return
        t = tensor.contiguous()
        # tensor 由 ViT/merger kernel 在计算流上异步写出, 而 put 是
        # SDMA 引擎直读 NPU 显存, 两者分属不同硬件队列、API 内部不与
        # torch 流同步 (对照 kv_pool pool_worker 先 record event 再
        # put 的模式)。发 put 前必须先等计算流完成, 否则 SDMA 可能读到
        # 未写完的脏数据存进 memcache, 之后所有命中该 key 的请求都会
        # 拿到损坏的 embedding。同时保证同步返回前 t 一直存活, 避免
        # contiguous() 临时张量被释放后底层显存被复用。
        torch.npu.current_stream().synchronize()
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
