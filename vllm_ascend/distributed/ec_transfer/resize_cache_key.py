# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""自定义 resize 缓存的 key 构建。

key 在 hash 之前由 3 部分组成:
  1. resize 图片得到的张量 (pixel_values 字节内容);
  2. 模型信息 (与 mm/kv cache 的 key 混入模型信息的方式一致:
     通过 MultiModalHasher.hash_kwargs 的 model_id 成分);
  3. 字面字符串 "resize_cache", 表明缓存类型。
三部分一起做一次 hash。
"""

import torch
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalKwargsItem

RESIZE_CACHE_TAG = "resize_cache"

# 与 vllm mm 配置默认算法一致; FIPS 部署可换 "sha256"
_HASH_ALGORITHM = "blake3"


def build_resize_cache_key(resized_tensor: torch.Tensor, model_id: str) -> str:
    """由 (resize 后张量, 模型信息, 类型标记) 生成缓存 key。"""
    return MultiModalHasher.hash_kwargs(
        model_id=model_id,           # 成分 2: 模型信息
        cache_type=RESIZE_CACHE_TAG,  # 成分 3: 缓存类型标记
        image=resized_tensor,         # 成分 1: resize 后的张量
    )


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
