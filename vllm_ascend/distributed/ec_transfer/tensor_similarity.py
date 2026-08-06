# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""resized 张量相似性比较工具 (调试用)。

用于 ECMemcacheConnector 在 scheduler 侧对新到请求的 resized 张量
与历史缓存的张量做两两比较, 辅助判断"不同图片 resize 后是否足够
相似" (例如验证 resize 缓存 key 的区分度 / 排查误命中)。

度量指标:
  - 余弦相似度: 方向一致性, 对幅度差异不敏感;
  - 相对误差: |a - b| / (|b| + eps), 报 max 与 mean;
  - top-K 误差明细: 按绝对误差降序的前 K 个元素 (索引/两值/abs/rel),
    用于定位 max_rel_err 是被个别离群点顶高还是整体性偏差
    (eps 兜底下分母近 0 的元素会产生虚高的 rel_err, 必须看明细)。

形状不一致的两张量不做数值比较 (逐元素误差无意义), 仅记录形状。
"""

from dataclasses import dataclass, field

import torch

# 相对误差分母保护, 避免除零
_REL_ERR_EPS = 1e-8

# 误差明细输出的条目数
_TOP_K_ERRORS = 20


@dataclass
class TensorErrorEntry:
    """单个元素的误差明细; index 为原形状下的坐标。"""

    index: tuple[int, ...]
    val_a: float
    val_b: float
    abs_err: float
    rel_err: float

    def format(self) -> str:
        return (
            f"    idx={self.index} a={self.val_a:+.6e} b={self.val_b:+.6e} "
            f"abs_err={self.abs_err:.4e} rel_err={self.rel_err:.4e}"
        )


@dataclass
class TensorSimilarity:
    """一对张量的相似性结果; comparable=False 时各指标为 None。"""

    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    comparable: bool
    cos_sim: float | None = None
    max_rel_err: float | None = None
    mean_rel_err: float | None = None
    top_errors: list[TensorErrorEntry] = field(default_factory=list)

    def format(self) -> str:
        if not self.comparable:
            return f"shape mismatch {self.shape_a} vs {self.shape_b}, skipped"
        return (
            f"cos_sim={self.cos_sim:.6f} "
            f"max_rel_err={self.max_rel_err:.4e} "
            f"mean_rel_err={self.mean_rel_err:.4e}"
        )

    def format_top_errors(self) -> str:
        """top-K 误差明细, 每元素一行 (按绝对误差降序)。"""
        return "\n".join(e.format() for e in self.top_errors)


def _unravel_index(flat_idx: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """flat 索引 → 原形状坐标 (等价 np.unravel_index, C 序)。"""
    coords = []
    for dim in reversed(shape):
        coords.append(flat_idx % dim)
        flat_idx //= dim
    return tuple(reversed(coords))


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> TensorSimilarity:
    """比较两个张量的相似性 (余弦相似度 + 相对误差 + top-K 误差明细)。

    形状一致时逐元素比较; 不一致时仅返回形状信息, 不做数值比较。
    统一转 float32 再计算, 避免 bf16/fp16 下的精度噪声。
    """
    shape_a = tuple(a.shape)
    shape_b = tuple(b.shape)
    if shape_a != shape_b:
        return TensorSimilarity(
            shape_a=shape_a, shape_b=shape_b, comparable=False
        )

    fa = a.detach().float().flatten()
    fb = b.detach().float().flatten()

    cos_sim = torch.nn.functional.cosine_similarity(fa, fb, dim=0).item()
    abs_err = (fa - fb).abs()
    rel_err = abs_err / (fb.abs() + _REL_ERR_EPS)

    k = min(_TOP_K_ERRORS, abs_err.numel())
    _, top_flat_idx = abs_err.topk(k)
    top_errors = [
        TensorErrorEntry(
            index=_unravel_index(int(i), shape_a),
            val_a=fa[i].item(),
            val_b=fb[i].item(),
            abs_err=abs_err[i].item(),
            rel_err=rel_err[i].item(),
        )
        for i in top_flat_idx.tolist()
    ]

    return TensorSimilarity(
        shape_a=shape_a,
        shape_b=shape_b,
        comparable=True,
        cos_sim=cos_sim,
        max_rel_err=rel_err.max().item(),
        mean_rel_err=rel_err.mean().item(),
        top_errors=top_errors,
    )
