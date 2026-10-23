# L2 pHash 优化与新版接口适配

日期：2026-10-22。

本次实现完整 pHash 直接检索与 8 band 倒排索引共用一份 resized key
映射，并使 pHash、SSIM 和 resized tensor 传输跟随输入设备。

本次迁移基于 L2 提交 `5fecef105`，保留 `956d74c83` 中的 pHash 检索优化。

## 1. 修改范围

生产代码位于 `vllm_ascend/distributed/ec_transfer/`：

| 文件 | 修改 |
| --- | --- |
| `perceptual_ssim_cache_matcher.py` | 增加 band 索引、零距离直查、复用批量检索、等待并行任务完成、设备输入留在调用线程、补齐最终 SSIM 阈值检查 |
| `phash.py` | 输入设备上 FP32 计算、按设备缓存 DCT 常量、打包 8 字节结果、使用 `int.bit_count()` |
| `ssim.py` | 输入设备上 FP32 计算、按设备缓存高斯核、拒绝不同设备之间隐式比较 |
| `memcache_resized_tensor_store.py` | 去掉强制 CPU 搬移、匹配传输方向与缓冲区设备、保护外部拷贝和缓冲区复用、串行化 PUT 注册事务 |

`ec_memcache_connector.py` 本身未修改。

## 检索行为

- `max_hamming == 0`：直接查询完整 pHash 字典，返回 key 列表副本。
  不扫描全表、不计算汉明距离、不维护 band 索引。
- `1 <= max_hamming <= 7`：查询八个 8-bit band 桶，合并候选后计算完整
  汉明距离，再按 pHash 数值顺序展开各自的 key 列表。
- 最多七个不同 bit 不会改变所有八个 band，因此该范围内不会因粗筛漏掉
  符合汉明距离条件的 pHash。完整汉明过滤及 SSIM 检查仍是必需步骤。
- 同一批次中的重复查询复用检索结果，同时保留原有结果重复次数和顺序。
- resized tensor 成功持久化后，在同一把 `_similarity_state_lock` 中发布
  完整映射、band 索引和 similarity cache，失败时不发布。

索引仅存在于 matcher 实例内，随实例销毁；没有新增跨进程共享、持久化或
过期清理。非零距离模式增加约八倍于不同 pHash 数量的集合成员；当图片的
pHash 高度相关时，候选数仍可能退化到全量，不承诺固定加速比。

## 新版 L2 接口

加载、批次构造和 SSIM 评分统一使用 `ResizedTensorLoadRequest` 的
`output_index`、`resized_key`、`nbytes` 字段。每个批次的本地输出索引重新
编号，传输接口仍按原有 key 顺序调用。

保留新版 connector 对空缺失列表的短路处理，以及统一 similarity 状态锁。
最终 SSIM 判定使用严格 `score > ssim_threshold`，与缓存查找和批次提前结束
一致；等于阈值和 NaN 均不作为最终命中。

每批计算后的累计结果按分数降序、同分按 key 排序。regular 批次出现合格
候选时跳过 oversized 加载。空分数列表不会调用 `max()`，后续批次仍可继续。

## 设备和并发

pHash 和 SSIM 在输入设备上使用 FP32。pHash 将 64 个 bit 打包为八字节再
返回 CPU，Python 索引仍在 CPU；不保证 CPU/NPU 浮点计算逐位一致。

| 缓冲区设备 | PUT | GET |
| --- | --- | --- |
| CPU | `COPY_H2G` | `COPY_G2H` |
| NPU | `COPY_L2G` | `COPY_G2L` |

接收缓冲区跟随输入设备，切换设备前注销旧缓冲区。NPU 在外部 SDK 读取
指针前及接收缓冲区 clone 完成后进行必要同步，异常路径也保护缓冲区复用。
PUT 的注册、拷贝、注销由实例锁保护，防止重复指针或共享注册状态并发冲突。

CPU 任务使用线程池，全部任务结束后再传播错误；设备张量保留在调用线程，
避免丢失调用方 NPU stream。没有新增 `tensor.item()` 热路径。
