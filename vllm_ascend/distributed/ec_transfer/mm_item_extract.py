# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""mm item 字段提取工具 (供 ECMemcacheConnector 的 L2 模糊匹配使用)。

L2 不做独立的 resize cache 存储: 相似图经 scheduler 侧的
phash_to_mm_hash 字典 (pHash → (mm_hash, grid_thw)) 直接复用候选图
的 L1 条目, 因此这里只保留从处理后的 mm item 中提取 resized 张量与
grid 信息的两个工具函数。
"""

import torch
from vllm.multimodal.inputs import MultiModalKwargsItem


def extract_resized_tensor(item: MultiModalKwargsItem) -> torch.Tensor | None:
    """从处理后的 mm item 中取 resize 后的张量 (Qwen-VL 系列为 pixel_values)。"""
    for field in ("pixel_values", "pixel_values_videos"):
        if field in item:
            data = item[field].data
            if isinstance(data, torch.Tensor):
                return data
    return None


def extract_image_grid(item: MultiModalKwargsItem) -> tuple[int, ...] | None:
    """从 mm item 中取图片的网格/宽高信息 (Qwen-VL 系列为 image_grid_thw)。

    调度阶段原始图片 (PIL/URL) 已不保留, 无法拿到文件名, 只能用
    grid_thw (t, h, w, patch 计数) 作为图片的尺寸标识。返回 None
    表示该 item 没有 grid 字段。
    """
    for field in ("image_grid_thw", "video_grid_thw"):
        if field in item:
            data = item[field].data
            if isinstance(data, torch.Tensor) and data.numel() >= 3:
                return tuple(int(x) for x in data.flatten()[:3].tolist())
    return None
