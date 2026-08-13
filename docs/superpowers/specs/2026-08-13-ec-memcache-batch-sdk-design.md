# EC Memcache Connector SDK 调用批量化改造设计

日期：2026-08-13
分支：hx（来源 `github.com/hanxi-java/vllm-ascend` fork）
关联代码：`vllm_ascend/distributed/ec_transfer/ec_memcache_connector.py`（唯一修改文件）

## 一、背景与目标

hx 分支基于 memcache 实现了 encoder embedding 缓存卸载与复用（`ECMemcacheConnector`，两级缓存：L1 精确 mm_hash 命中 + L2 pHash/SSIM 模糊复用）。

经对 SDK 源码（`C:\code\gitcode\memcache`，Python 绑定 `DistributedObjectStore`）逐层确认：SDK 在 C/C++/Python 各层均为批量接口（`batch_is_exist` / `batch_get_key_info` / `batch_get_into_layers` / `batch_put_from_layers`），接受 key 列表、按位返回结果、无批量大小硬上限（内部 `aggregate_num=122` 仅为 IO 分组参数）；vllm-ascend 的 `MemcacheBackend` 封装层本来就是 list 形接口。而 connector 在 5 处 for 循环中每次只传单元素列表 `[key]`，造成 O(N) 次 SDK 往返。

**目标**：将 5 处循环内的 SDK 调用全部改为批量调用，语义保持等价。

**约束**：
1. 本环境无 python/GPU，不在本地执行 python，功能验证在远程进行
2. 注释用中文，格式与现有注释一致（中文、重原理说明）；logger 日志保持英文原格式
3. 只修改 hanxi-java 引入的代码 —— 实际只改 `ec_memcache_connector.py` 一个文件（`memcache_backend.py` 的批量方法已存在，无需修改）

## 二、SDK 接口语义要点

`MemcacheBackend`（`memcache_backend.py`）现有接口，全部 list 形：

| 方法 | 底层 | 返回语义 |
|---|---|---|
| `exists(keys)` | `batch_is_exist` | 按位：1=存在 0=不存在 -1=错误 |
| `batch_get_key_info(keys)` | `batch_get_key_info` | 按位 `KeyInfo`；`size()==0` 表示不存在/已淘汰 |
| `get(keys, addr, size, direction=COPY_G2L)` | `batch_get_into_layers` | 按位错误码（0=成功） |
| `put(keys, addr, size, direction=COPY_L2G)` | `batch_put_from_layers` | 按位错误码（0=成功） |

- 结果按输入顺序按位返回，部分失败按 key 呈现
- `get`/`put` 的 `addr`/`size` 为双层列表：外层按 key、内层按 layer；单 buffer 用法为 `[[ptr1], [ptr2], ...]`
- 整批失败语义：`batch_get_key_info` 整批失败返回空；`get`/`put` 异常返回 None；任何非法 key（空/超 256 字节）会使 get/put 整批中止——本方案不改 key 生成逻辑（mm_hash 为内容哈希，恒 1-256 字节），无新风险

## 三、总体改造模式

统一模式：**循环外收集 keys → 批量调用 → 按位把结果映射回循环逻辑**。

辅助函数变更：

| 原函数 | 动作 | 说明 |
|---|---|---|
| `_resized_get` | 替换为 `_resized_get_multi(mm_hashes) -> list[Tensor\|None]` | 唯一调用点 `_ssim_score_candidates` 改为批量调用 |
| `_ec_get` | 替换为 `_ec_get_multi(keys) -> list[Tensor\|None]` | 唯一调用点 `start_load_caches` 改为批量调用 |
| `_ec_put` | 替换为 `_ec_put_multi(items: list[(key, Tensor)])` | 调用点 `start_load_caches`（循环）、`save_caches`（单元素列表）均改 |
| `_resized_put` | 保持不变（单 key） | 调用方是每图一个的 daemon 线程，天然单次 |
| `has_cache_item` | 保持不变 | 单次调用，无循环 |

save 路径不做"跨多次上游调用攒批"：攒批必须等下个 step 的 `start_load_caches` 才能 flush，会把 L1 写入推迟一个 step，且与 scheduler 下个 step 的 exists 检查存在跨进程竞态，相邻 step 同图请求会多算一次 ViT；save 每次 miss 才发生一次、不在本文件可控的循环内，不是热点，故 `save_caches` 用单元素列表调用 `_ec_put_multi` 保持同步写语义。

## 四、逐处改造设计

### ① `ensure_cache_available`（scheduler，请求内多图片 exists）

两段式：先收集待查 identifier（`seen` 集对重复项去重——原逐条路径中重复项在首次 exists 命中后被 `_mm_hash_hits` 跳过，收集期等价跳过，避免重复计数），一次 `exists(keys)`；再按位处理：命中走原 L1 分支（登记/计数/日志），未命中走原 L2 分支（`extract_resized_tensor` + `_l2_lookup`）。顺序由"逐条 exists→L2 交错"变为"全部 exists 先完成、再逐条 L2"——安全依据：`_l2_lookup` 只读 phash 索引/resized 注册表，不读 `_mm_hash_hits`，feature 间无依赖。

### ② `_similarity_lookup`（scheduler，pHash 候选 exists）

`_phash_candidates` 返回已物化的 list（无惰性求值问题）。一次 `exists(全部候选)`，按序取第一个 `== 1` 者命中——"首个通过 exists 确认者命中"语义保留，只是对首个命中之后的候选也多发了只读查询（无副作用，换一次网络往返覆盖全部候选）。

### ③ `_ssim_score_candidates`（scheduler，最大热点：SSIM 候选批量读回）

循环前一次 `_resized_get_multi(candidate_mm_hashes)` 批量取回全部候选 resized 张量（内部一次 `batch_get_key_info` + 一次 `get`，**分块执行**见配置节），单独计时并打一条汇总日志 `"%s BATCH-GET: candidates=%d elapsed=%.3fms"`；循环内按位取张量做灰度平面 + SSIM，原日志/统计/阈值逻辑不变。

**计时口径变化（有意为之）**：原逐候选 `elapsed` 含 memcache 读回耗时；批量化后读回发生在循环外单独计时，`elapsed` 变为纯 SSIM 计算耗时（`_ssim_cmp_total_ms` 累计口径随之变化），docstring 中说明。

### ④ `_first_existing_candidate`（scheduler，SSIM 胜者 exists）

得分降序排序后一次 `exists(全部候选)`，按序取第一个 `== 1` 返回；语义完全一致（早期退出保留）。

### ⑤ `start_load_caches`（worker，每步必走的主循环）

三段式：
- ① 收集：过滤 `encoder_cache` 已含条目；`seen_cur` 集对重复 current 去重（原路径重复 current 在首次注入后被 encoder_cache 命中跳过）；`unique_hits` 对取数键去重（原路径重复取数键重复拉取，批量后一次拉取、同一张量共享注入多个 current——下游只读，安全）
- 一次 `_ec_get_multi(unique_hits)` → `hit_to_embedding` 映射
- ② 注入：按位取 embedding，None 走原警告日志；否则注入 `encoder_cache[current]`；L2 模糊命中（`hit != current`）且 rank 0 时收集回填对
- ③ 回填：一次 `_ec_put_multi(backfills)`（exists 去重 + 单次同步 + 单次 put）。回填 key 与取数 key 无重叠（回填只发生在 `hit != current` 的条目，这类 current 从未作为 hit 被读过），顺序无耦合

日志顺序由"逐条 LOAD/PUT 交错"变为"全部 LOAD 后 PUT"，仅观感变化。

### 辅助函数内部逻辑

`_resized_get_multi`（CPU 目标，scheduler 侧）：
- 空列表直接返回 `[]`；按 `self._resized_get_chunk` 分块
- 块内先过滤无本地元信息（`_mm_hash_to_resized_meta`）的 key；一次 `batch_get_key_info`（整批返回空 → 防御按全缺失 + warning `"EC RESIZED BATCH-GET key_info failed"`）
- 按位：`size()==0` 跳过（已淘汰）；按各自 `(shape, dtype)` 分配 CPU buffer；`buf.nbytes != nbytes` → 原 size mismatch 警告并跳过
- 有效子集一次 `get(keys, [[ptr]], [[nbytes]], MmcDirect.COPY_G2H.value)`；按位 `res[i] != 0` → 原失败警告
- 结果按原位置写回（`(chunk 内下标, get 批内下标, buf)` 显式簿记）；目标为 CPU 内存，无需 `npu.synchronize`

`_ec_get_multi`（NPU 目标，worker 侧）：
- 空列表直接返回 `[]`；一次 `batch_get_key_info(keys)`（整批返回空 → 防御按全缺失 + warning `"EC memcache batch_get_key_info failed"`）
- 按位：`size()==0` → None；`num_tokens = nbytes // elem_size // hidden_dim` 分配 NPU buffer
- 有效子集一次 `get(keys, [[ptr]], [[nbytes]])`（默认 COPY_G2L）；按位 `res[i] != 0` → 原失败警告
- 整批一次 `torch.npu.synchronize()`（SDMA 直写 buf 与 forward 计算流分属不同硬件队列，注入前必须同步；单 key 路径逐次同步，批量路径整批一次）

`_ec_put_multi`（worker 侧）：
- 空列表直接返回；一次 `exists(keys)` 批量去重（返回空 → 防御按全部不存在）
- 对不存在的条目 `t = tensor.contiguous()` 收集进 `to_put`（临时张量持有在列表中，保证存活到 put 返回——SDMA 直读）
- 一次 `torch.npu.current_stream().synchronize()`（ViT/merger kernel 计算流异步写出、put 走 SDMA 直读，不先同步会存脏数据污染所有命中该 key 的请求）
- 一次 `put(keys, [[t.data_ptr()]], [[t.nbytes]])`（默认 COPY_L2G）；逐 key 失败结果忽略（与原单 key 路径一致，backend 内部有日志上报）

### 批量生效观测日志

批量成功路径新增 3 条 info 日志（中文注释、英文日志，风格与现有一致），任何模式（含纯 L1）下均可据此确认批量生效：

| 位置 | 日志 |
|---|---|
| `ensure_cache_available` 批量 exists 后 | `EC BATCH EXISTS: keys=%d` |
| `_ec_get_multi` 批量 get 后 | `EC BATCH GET: keys=%d` |
| `_ec_put_multi` 批量 put 前 | `EC BATCH PUT: keys=%d` |

旧代码为逐条单 key 调用、不存在这些日志行；日志中 keys 数即批量规模。空集合入口不发 SDK 调用，也不打这些日志。

## 五、配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `EC_RESIZED_GET_CHUNK` | `32` | `_resized_get_multi` 的批读分块大小。单张 resized 张量（pixel_values）可达 ~20MB，一次性读回全部候选会把 scheduler 进程 CPU 峰值内存推高数 GB（ssim matcher 的候选集可能是整个同形状注册表），故按块批读；SDK 对 batch 大小无硬上限，分块纯粹为约束本地内存峰值（默认 32 块峰值约 640MB，内存紧张可调小）。读取方式与 `EC_L2_MAX_HAMMING` / `EC_SSIM_MIN_SCORE` 等同款（`__init__` SCHEDULER 分支 `os.getenv`） |

## 六、边角情况

| 情况 | 处理 |
|---|---|
| 空集合 | 全部批量入口判空，不发 SDK 调用 |
| 重复 key | ① `seen` 去重 identifier；⑤ `seen_cur` + `unique_hits` 去重；②④ 不去重（重复实际不可能：确定性 pHash、索引先登记者赢） |
| SDK 整批失败（返回空/None） | 防御按全 miss / 全不存在处理，与原单 key 路径语义一致；新增 2 条 warning + 1 条批量读回汇总日志 |
| 批量 get 部分淘汰/失败 | 按位 None + 原 warning 日志格式 |
| 早期退出语义（②④） | 批量 exists 后按原顺序扫描取首个命中，完全等价 |
| torch 同步次数 | ⑤ 原来每 get 一次 `npu.synchronize`、每 put 一次 stream sync；批量化后整步各一次 |

## 七、验证

静态验证（本环境无 python）：
1. `git diff` 审查：仅 `ec_memcache_connector.py` 被修改
2. grep：`_ec_get(`/`_ec_put(`/`_resized_get(` 单 key 旧名零残留；`self._backend.exists(` 共 6 处（3 个批量点 + `_resized_put` + `has_cache_item` + `_ec_put_multi`）
3. 按位映射核对：所有批量结果经 `zip` 同长度消费；get 批内下标用显式簿记
4. torch 同步核对：`_ec_get_multi` 内一次 `npu.synchronize`；`_ec_put_multi` 内一次 `current_stream().synchronize`；`_resized_get_multi` 无 sync（CPU 目标）
5. 行宽 ≤ 120；全部既有日志格式字符串保持原样

远程功能验证建议：跑原有测试场景（L1 命中 / L2 phash 命中 / ssim 命中 / miss 回填 / 相同图多次请求），对照日志确认：命中路径与改造前一致、`EC meta` 命中计数一致、SDK 往返次数显著下降（出现 `EC SSIM BATCH-GET` 等批量日志）。
