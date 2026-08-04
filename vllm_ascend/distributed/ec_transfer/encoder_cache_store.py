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

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm_ascend.distributed.ec_transfer.ec_memcache_backend import EcMemcacheBackend
from vllm_ascend.distributed.ec_transfer.ec_store_client import (
    get_zmq_rpc_path_ec_lookup,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class EncoderCacheStore:
    """Worker-side encoder cache store backed by memcache.

    Mirrors the KV-transfer backend: registers buffers with HYBM before
    every operation, then uses batch_put_from_layers / batch_get_into_layers
    with explicit L2G/G2L directions.
    """

    def __init__(self, vllm_config: "VllmConfig", local_rank: int):
        self._store = EcMemcacheBackend(local_rank)
        model_config = vllm_config.model_config
        self._model_name = model_config.model.split("/")[-1]
        self._hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
        self._dtype = model_config.dtype
        self._elem_size = torch.tensor([], dtype=self._dtype).element_size()

        # ZMQ REP server for scheduler exists queries
        self._running = True
        socket_path = get_zmq_rpc_path_ec_lookup(vllm_config)
        self._zmq_ctx = zmq.Context()  # type: ignore[attr-defined]
        self._zmq_socket = make_zmq_socket(
            self._zmq_ctx, socket_path, zmq.REP, bind=True  # type: ignore[attr-defined]
        )
        self._zmq_thread = threading.Thread(target=self._zmq_loop, daemon=True)
        self._zmq_thread.start()
        logger.info(
            "EncoderCacheStore started on %s (model=%s hidden_dim=%d)",
            socket_path,
            self._model_name,
            self._hidden_dim,
        )

    # ---- NPUModelRunner calls ----

    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        """Store *tensor* as the encoder output for *mm_hash*."""
        key = self._make_key(mm_hash)
        self._store.put(key, tensor)

    def get(self, mm_hash: str) -> torch.Tensor | None:
        """Return the cached encoder output for *mm_hash*, or ``None``."""
        key = self._make_key(mm_hash)
        return self._store.get(key, self._elem_size, self._hidden_dim, self._dtype)

    # ---- ZMQ server ----

    def _zmq_loop(self) -> None:
        """ZMQ REP loop: receive mm_hash, reply ``b'1'`` (exists) or ``b'0'``."""
        while self._running:
            try:
                mm_hash = self._zmq_socket.recv_string()
                key = self._make_key(mm_hash)
                exists = self._store.exists([key])[0] == 1
                self._zmq_socket.send(b"1" if exists else b"0")
            except zmq.error.ZMQError:
                break

    # ---- internal helpers ----

    def _make_key(self, mm_hash: str) -> str:
        return f"{self._model_name}@cache_role:ec@{mm_hash}"


def _get_encoder_cache_hidden_dim(vllm_config: "VllmConfig") -> int:
    """Return per-token hidden dimension for encoder cache entries.

    Mirrors ``ec_connector/cpu/common.py:38-59`` (Qwen3-VL deepstack support).
    """
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None) if hf_config else None
    if vision_config is not None:
        out_hidden_size = getattr(vision_config, "out_hidden_size", None)
        deepstack_indexes = getattr(
            vision_config, "deepstack_visual_indexes", None
        )
        if out_hidden_size is not None and deepstack_indexes:
            return out_hidden_size * (1 + len(deepstack_indexes))
    return model_config.get_inputs_embeds_size()
