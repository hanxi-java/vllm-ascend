"""ECMemcacheConnector — 基于 memcache 的 encoder cache connector。

通过 ec_connector 框架接入 vLLM (无需修改 vllm 仓库):
    --ec-transfer-config '{
        "ec_connector": "ECMemcacheConnector",
        "ec_connector_module_path":
            "vllm_ascend.distributed.ec_transfer.ec_memcache_connector",
        "ec_role": "ec_both"
    }'

缓存命中规则 (逻辑平移自 MultiLevelEncoderCacheManager):
  - key = mm_hash (request.mm_features[i].identifier), 命中即跳过 ViT;
  - 未命中才真正执行 ViT, 算完后以本图 mm_hash 回填 memcache。

淘汰由 memcache 组件自治, connector 不实现任何驱逐逻辑。
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import memcache_hybrid
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

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


@dataclass
class ECMemcacheConnectorMetadata(ECConnectorMetadata):
    """Per-step scheduler → worker payload.

    loads: 本步缓存命中、需要 worker 从 memcache 加载的 mm_hash 列表。
           worker 取出后以 mm_hash 注入 encoder_cache。
    saves: 本步未命中、需要 worker 在 ViT 算完后回填 (key=mm_hash)
           的 mm_hash 集合。
    """

    loads: list[str] = field(default_factory=list)
    saves: set[str] = field(default_factory=set)


@dataclass
class _GetRequestParam:
    """_ec_get_batch 中的一个读回条目: 把一次 get 所需的字段绑定在一起。

    一个对象对应一个 alive key (未淘汰、待读回的条目), 自包含全部信息,
    不再依赖"多个并行列表按下标对齐":
      - src_idx: 该 key 在原始 keys 列表里的下标 (映射回结果用);
      - key: memcache key;
      - addr / size: 读回 buffer 地址与字节数 (单层 list);
      - buf: 分配的 NPU buffer。
    """

    src_idx: int
    key: str
    addr: list[int]
    size: list[int]
    buf: torch.Tensor


class ECMemcacheConnector(ECConnectorBase):
    """Encoder cache connector backed by memcache."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        model_config = vllm_config.model_config

        self._model_id: str = model_config.model

        if role == ECConnectorRole.SCHEDULER:
            # 元数据面 client: exists 查询, 不申请存储介质
            self._backend = MemcacheBackend.create_scheduler_client(
                vllm_config.parallel_config
            )
            # 预计算的命中结果 (ensure_cache_available 填充):
            # L1 命中的 identifier 集合
            self._mm_hash_hits: set[str] = set()
            # 累计统计
            self._full_pixel_hit_count = 0
            self._miss_count = 0
            # ── 本步的 loads/saves 登记簿 (build_connector_meta 打包下发后清空) ──
            #
            # _mm_hashes_need_loads: 本步缓存命中、需要 worker 从 memcache
            #   加载的条目 (mm_hash 列表)。
            self._mm_hashes_need_loads: list[str] = []
            #
            # _mm_hashes_need_saves: 本步未命中、需要 worker 在 ViT 算完后
            #   回填 memcache 的 mm_hash 集合。worker save_caches 回调时
            #   据 mm_hash 从 encoder_cache 取刚算出的 embedding 写 L1
            #   (key=mm_hash)。
            self._mm_hashes_need_saves: set[str] = set()
        elif role == ECConnectorRole.WORKER:
            # 数据面 client: embedding 的实际读写
            self._backend = MemcacheBackend(vllm_config.parallel_config)
            self._hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
            self._dtype = model_config.dtype
            self._elem_size = torch.empty(0, dtype=self._dtype).element_size()
            # TP 下各 rank 的 embedding 相同 (ViT 输出 all-reduce),
            # 写操作只由 rank 0 执行, 读操作各 rank 独立进行
            self._save_rank = get_world_group().rank == 0
            # 落盘线程池: 线程数 = world_size * 2, 复用线程避免每步新建线程
            self._put_executor = ThreadPoolExecutor(
                max_workers=get_world_group().world_size * 2,
                thread_name_prefix="ec-put",
            )
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

        logger.info(
            "ECMemcacheConnector init: role=%s model=%s",
            role,
            self._model_id,
        )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        """调度前预计算本请求全部 mm 条目的 L1 命中情况。
        """
        # 先收集本步全部待查 identifier, 一次性批量 exists (结果位置与
        # 收集顺序一一对应), 再逐个处理 L1 命中登记。同一 identifier 只
        # 查一次: 原逐条路径中重复项在首次 exists 命中后被 _mm_hash_hits
        # 跳过, 这里用 seen 列表在收集期等价跳过, 避免批量结果里对重复键
        # 重复计数。
        seen_mm_hashes_in_current_request: list[str] = []
        for feature in request.mm_features:
            current_image_mm_hash = feature.identifier
            if (current_image_mm_hash in self._mm_hash_hits
                    or current_image_mm_hash in seen_mm_hashes_in_current_request):
                continue
            seen_mm_hashes_in_current_request.append(current_image_mm_hash)

        exists_res = (
            self._backend.exists(seen_mm_hashes_in_current_request)
            if seen_mm_hashes_in_current_request else []
        )
        if not exists_res and seen_mm_hashes_in_current_request:
            # 防御: SDK 异常返回空时按全部未命中处理 (与单 key 路径把
            # exists != 1 当未命中的语义一致)
            exists_res = [0] * len(seen_mm_hashes_in_current_request)
        if seen_mm_hashes_in_current_request:
            logger.debug("EC BATCH EXISTS: keys=%d", len(seen_mm_hashes_in_current_request))
        for current_image_mm_hash, existed in zip(
                seen_mm_hashes_in_current_request, exists_res):
            # L1: key = identifier (mm_hash)
            if existed == 1:
                self._mm_hash_hits.add(current_image_mm_hash)
                self._full_pixel_hit_count += 1
                logger.debug("EC FULL-PIXEL HIT (sched): current mm_hash=%s", current_image_mm_hash)
        # 不做延迟调度 (memcache 查询是同步的, 结果立即可用)
        return True

    def has_cache_item(self, identifier: str) -> bool:
        if identifier in self._mm_hash_hits:
            return True
        # ensure_cache_available 未覆盖的场景 (如 running 请求的 chunk 续调度)
        # 兜底一次直接 L1 查询
        if self._backend.exists([identifier]) == [1]:
            self._mm_hash_hits.add(identifier)
            self._full_pixel_hit_count += 1
            logger.debug("EC FULL-PIXEL HIT (sched, fallback): mm_hash=%s", identifier)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """命中项登记 load, 未命中项登记 save (两类条目 scheduler 都会调到)。"""
        feature = request.mm_features[index]
        current_image_mm_hash = feature.identifier

        # L1 命中: 登记 load
        if current_image_mm_hash in self._mm_hash_hits:
            self._mm_hashes_need_loads.append(current_image_mm_hash)
            return

        # 未命中: ViT 将由 worker 执行, 登记 L1 回填
        self._miss_count += 1
        self._mm_hashes_need_saves.add(current_image_mm_hash)

    @property
    def hit_rate(self) -> float:
        """累计命中率: L1 命中数 / 总判定数。"""
        hits = self._full_pixel_hit_count
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
            logger.debug(
                "EC meta: %d loads, %d saves this step | "
                "EC meta loads: %r, EC meta saves: %r this step | "
                "full_pixel_hits=%d misses=%d hit_rate=%.2f%%",
                len(meta.loads),
                len(meta.saves),
                meta.loads,
                meta.saves,
                self._full_pixel_hit_count,
                self._miss_count,
                self.hit_rate * 100,
            )
        # 每步重建, 同时清空跨步状态 (累计统计字段保留)
        self._mm_hashes_need_loads = []
        self._mm_hashes_need_saves = set()
        self._mm_hash_hits.clear()
        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMemcacheConnectorMetadata)
        loading_hashes_in_current_step: list[str] = []
        seen_hashes_in_current_step: set[str] = set()
        for current_image_mm_hash in metadata.loads:
            if (current_image_mm_hash in encoder_cache
                    or current_image_mm_hash in seen_hashes_in_current_step):
                continue
            seen_hashes_in_current_step.add(current_image_mm_hash)
            loading_hashes_in_current_step.append(current_image_mm_hash)
        if not loading_hashes_in_current_step:
            return

        embeddings = self._ec_get_batch(loading_hashes_in_current_step)
        for current_image_mm_hash, embedding in zip(loading_hashes_in_current_step, embeddings):
            if embedding is None:
                logger.warning(
                    "EC LOAD miss: current_image_mm_hash=%s (may be evicted by memcache?)",
                    current_image_mm_hash,
                )
                continue
            encoder_cache[current_image_mm_hash] = embedding
            logger.debug(
                "EC LOAD: current_image_mm_hash=%s embedding shape=%r",
                current_image_mm_hash,
                embedding.shape,
            )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """ViT 算完后回填 L1 (key=mm_hash)。"""
        if not self._save_rank:
            return
        if mm_hash not in encoder_cache:
            return
        embedding = encoder_cache[mm_hash]
        self._ec_put(mm_hash, embedding)

    # ==============================
    # Worker-side memcache helpers
    # ==============================

    def _ec_get_batch(self, keys: list[str]) -> list[torch.Tensor | None]:
        """按 keys 批量读回 embedding (NPU 张量), 结果与入参位置一一对应。

        条目缺失 (key_info.size()==0) / 读失败的 key 对应 None。读回统一走
        G2L (直写本地设备内存)。批量 get 返回后统一同步一次计算流, 等价于
        原单 key 路径的"注入前同步一次" (SDMA 直写 buf 与计算流读分属不同
        硬件队列, 见下方注释)。
        """
        if not keys:
            return []
        key_infos = self._backend.batch_get_key_info(keys)
        logger.warning("EC memcache batch_get_key_info content: key_infos=%r, keys=%r", key_infos, keys)
        if not key_infos:
            # 整批查询失败 (SDK 返回空), 按全缺失处理
            logger.warning("EC memcache batch_get_key_info failed: keys=%s", keys)
            return [None] * len(keys)
        # 逐个 key 分配 NPU buffer (尺寸由 key_info 给出); 已淘汰 (size 0) 的
        # key 不入批。每个 alive key 封装成一个自包含的读回请求。
        request_params: list[_GetRequestParam] = []
        for i, (key, ki) in enumerate(zip(keys, key_infos)):
            if ki.size() == 0:
                continue
            nbytes = ki.size()
            num_tokens = nbytes // self._elem_size // self._hidden_dim
            buf = torch.empty(num_tokens, self._hidden_dim,
                              dtype=self._dtype, device="npu")
            request_params.append(_GetRequestParam(i, key, [buf.data_ptr()], [nbytes],
                                        buf))
        if not request_params:
            return [None] * len(keys)

        embeddings: list[torch.Tensor | None] = [None] * len(keys)
        group_res = self._backend.get(
            [req.key for req in request_params],
            [req.addr for req in request_params],
            [req.size for req in request_params],
            memcache_hybrid.G2L,
        )
        logger.debug("EC BATCH GET: keys length=%d keys content=%r direction=%d",
                     len(request_params), request_params, memcache_hybrid.G2L)
        if group_res is None:
            # SDK 异常返回 None: 全部按失败处理
            for req in request_params:
                logger.error("EC memcache get failed: key=%s res=None",
                               req.key)
            return embeddings
        for req, code in zip(request_params, group_res):
            if code != 0:
                logger.error("EC memcache get failed: key=%s res=%s",
                               req.key, code)
                continue
            embeddings[req.src_idx] = req.buf
            logger.debug("EC memcache get success: key=%s buf shape=%r",
                         req.key, req.buf.shape)
        # get 是 SDMA 引擎直写 buf, 与后续 forward 读 buf 的计算流分属
        # 不同硬件队列; 注入 encoder_cache 前同步一次, 保证计算流读到
        # 完整的拷贝结果, 否则模型拿到的是未写完的 buffer (垃圾输出)。
        # 单 key 路径逐次同步, 批量路径整批同步一次。
        # torch.npu.synchronize()
        return embeddings

    def _ec_put(self, mm_hash: str, embedding: torch.Tensor) -> None:
        """把单个 embedding 写回 memcache (key=mm_hash), 落盘在线程池完成。

        写前在主线程等计算流完成 (current_stream 是线程局部的, 必须在主线程
        同步到正确的计算流), 再把阻塞的 SDK 拷贝提交到线程池, 不阻塞调度
        主流程。
        """
        # 先 contiguous (若非连续会 enqueue 一次拷贝 kernel), 再同步, 保证
        # 线程池拿到 t 时已完整写入 (ViT/merger 的写出 + 可能的 contiguous
        # 拷贝)。t 由提交的任务持有引用, 在 put 返回前其存储不会被释放。
        t = embedding.contiguous()
        # tensor 由 ViT/merger kernel 在计算流上异步写出, 而 put 是
        # SDMA 引擎直读 NPU 显存, 两者分属不同硬件队列、API 内部不与
        # torch 流同步。发 put 前必须先等计算流完成, 否则 SDMA 可能读到
        # 未写完的脏数据存进 memcache, 之后所有命中该 key 的请求都会
        # 拿到损坏的 embedding。
        torch.npu.current_stream().synchronize()
        self._put_executor.submit(self._put_async, mm_hash, t)

    def _put_async(self, mm_hash: str, t: torch.Tensor) -> None:
        """线程池任务: 把已就绪的 tensor 写入 memcache, 异常只打日志不回传。

        写失败只影响缓存命中率 (该 key 后续仍走 miss 重算), 不影响正确性。
        """
        try:
            # 线程池线程需要设置自己的 device 上下文 (当前设备是线程局部的)
            torch.npu.set_device(t.device)
            self._backend.put([mm_hash], [[t.data_ptr()]], [[t.nbytes]])
            logger.debug("EC PUT: key=%s nbytes=%d", mm_hash, t.nbytes)
        except Exception:
            logger.exception("EC PUT failed: mm_hash=%s", mm_hash)

    def shutdown(self) -> None:
        """进程退出时关闭落盘线程池, 等待未完成的写入完成。"""
        executor = getattr(self, "_put_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)


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
