# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import ctypes
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.ec_transfer import memcache_resized_tensor_store as store_module
from vllm_ascend.distributed.ec_transfer.memcache_resized_tensor_store import (
    MemcacheResizedTensorStore,
    ResizedTensorLoadRequest,
)
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MmcDirect


def _store():
    backend = MagicMock()
    backend.exists.return_value = [0]
    backend.put_from.return_value = 0
    return MemcacheResizedTensorStore(backend, 2, "test-model", MIN_RESIZED_IMAGE_RECV_BUFFER_SIZE)


def test_receive_buffer_stays_on_query_device_and_is_reused():
    store = _store()
    try:
        buffer = store._ensure_recv_buffer(torch.device("cpu"))
        assert buffer.device.type == "cpu"
        assert store._ensure_recv_buffer(torch.device("cpu")) is buffer
        store._backend.register_buffer.assert_called_once_with([buffer.data_ptr()], [buffer.nbytes])
    finally:
        store.shutdown()


def test_cpu_receive_loads_real_bytes_with_host_direction():
    store = _store()
    current = torch.zeros(16, 12)
    expected = torch.arange(current.numel(), dtype=current.dtype).reshape_as(current)
    recv = torch.empty(current.nbytes, dtype=torch.uint8)
    outputs = [None]

    def copy_into(keys, addresses, sizes, direction):
        assert direction == MmcDirect.COPY_G2H.value
        ctypes.memmove(addresses[0], expected.data_ptr(), sizes[0])
        return [0]

    store._backend.batch_get_into_buffers.side_effect = copy_into
    try:
        store._load_batch_into_buffer([ResizedTensorLoadRequest(0, "candidate", current.nbytes)], outputs, recv, current)
        assert torch.equal(outputs[0], expected)
        recv.zero_()
        assert torch.equal(outputs[0], expected), "outputs must survive receive-buffer reuse"
    finally:
        store.shutdown()


def test_npu_put_keeps_device_tensor_and_waits_before_external_copy():
    store = _store()
    tensor = MagicMock()
    tensor.device = SimpleNamespace(type="npu")
    tensor.dtype = torch.float32
    tensor.shape = (16, 12)
    tensor.nbytes = 768
    tensor.detach.return_value.contiguous.return_value = tensor
    tensor.detach.return_value.to.return_value.contiguous.return_value = tensor
    tensor.data_ptr.return_value = 1024
    stream = MagicMock()
    operations = []
    stream.synchronize.side_effect = lambda: operations.append("ready")
    store._backend.put_from.side_effect = lambda *args: operations.append("put") or 0
    store._backend.unregister_buffer.side_effect = lambda *args: operations.append("unregister")
    npu = SimpleNamespace(current_stream=MagicMock(return_value=stream))
    try:
        with patch.object(store_module.torch, "npu", npu, create=True):
            assert store.put("image", tensor, (1, 4, 4))
        tensor.detach.return_value.to.assert_not_called()
        assert store._backend.put_from.call_args.args[-1] == MmcDirect.COPY_L2G.value
        assert operations == ["ready", "put", "unregister"]
    finally:
        store.shutdown()


def test_npu_receive_selects_device_direction_and_completes_copy_out():
    store = _store()
    recv = MagicMock()
    recv.device = SimpleNamespace(type="npu")
    recv.data_ptr.return_value = 4096
    current = torch.zeros(16, 12)
    operations = []
    stream = MagicMock()
    stream.synchronize.side_effect = lambda: operations.append("ready")
    store._backend.batch_get_into_buffers.side_effect = lambda *args: operations.append("get") or [0]
    npu = SimpleNamespace(current_stream=MagicMock(return_value=stream))
    try:
        with patch.object(store_module.torch, "npu", npu, create=True):
            store._load_batch_into_buffer([ResizedTensorLoadRequest(0, "candidate", current.nbytes)], [None], recv, current)
        assert store._backend.batch_get_into_buffers.call_args.args[-1] == MmcDirect.COPY_G2L.value
        assert operations[-1] == "ready", "device clones must finish before buffer reuse"
    finally:
        store.shutdown()


def test_parallel_put_serializes_registered_pointer_lifetimes():
    store = _store()
    tensor = torch.ones(16, 12)
    first_in_copy = Event()
    release_first = Event()
    second_entered_put = Event()
    second_registered = Event()
    state_lock = Lock()
    active = set()
    overlap = []

    def register(ptrs, sizes):
        with state_lock:
            overlap.append(bool(active))
            active.add(ptrs[0])
            if len(overlap) == 2:
                second_registered.set()

    def unregister(ptrs, sizes):
        with state_lock:
            active.discard(ptrs[0])

    def copy(key, *args):
        if key.startswith("resized_first_"):
            first_in_copy.set()
            assert release_first.wait(5)
        return 0

    def second_put():
        second_entered_put.set()
        return store.put("second", tensor, (1, 4, 4))

    store._backend.register_buffer.side_effect = register
    store._backend.unregister_buffer.side_effect = unregister
    store._backend.put_from.side_effect = copy
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(store.put, "first", tensor, (1, 4, 4))
            assert first_in_copy.wait(5)
            second = executor.submit(second_put)
            assert second_entered_put.wait(5)
            # Give an unprotected second registration time to overlap the
            # first transfer; with serialization it must remain blocked.
            overlapped = second_registered.wait(0.2)
            release_first.set()
            assert first.result(timeout=5)
            assert second.result(timeout=5)
        assert not overlapped
        assert overlap == [False, False]
        assert not active
    finally:
        release_first.set()
        store.shutdown()


def test_put_failure_unregisters_buffer():
    store = _store()
    store._backend.put_from.side_effect = RuntimeError("transfer failed")
    tensor = torch.ones(16, 12)
    try:
        with pytest.raises(RuntimeError, match="transfer failed"):
            store.put("image", tensor, (1, 4, 4))
        store._backend.unregister_buffer.assert_called_once_with([tensor.data_ptr()], [tensor.nbytes])
    finally:
        store.shutdown()


def test_receive_buffer_device_switch_unregisters_old_buffer_before_replacement():
    store = _store()
    old = MagicMock()
    old.device = SimpleNamespace(type="npu")
    old.data_ptr.return_value = 1234
    old.nbytes = store._recv_buffer_size
    store._recv_buffer = old
    store._recv_buffer_registered = True
    operations = []
    stream = MagicMock()
    stream.synchronize.side_effect = lambda: operations.append("ready")
    store._backend.unregister_buffer.side_effect = lambda *args: operations.append("unregister")
    store._backend.register_buffer.side_effect = lambda *args: operations.append("register")
    npu = SimpleNamespace(current_stream=MagicMock(return_value=stream))
    try:
        with patch.object(store_module.torch, "npu", npu, create=True):
            replacement = store._ensure_recv_buffer(torch.device("cpu"))
        assert replacement.device.type == "cpu"
        assert operations == ["ready", "unregister", "register"]
        store._backend.unregister_buffer.assert_called_once_with([1234], [old.nbytes])
    finally:
        store.shutdown()


def test_failed_npu_clone_finishes_reads_before_temporary_buffer_unregister():
    store = _store()
    current = torch.zeros(16, 12)
    recv = MagicMock()
    recv.device = SimpleNamespace(type="npu")
    recv.data_ptr.return_value = 4096
    recv.nbytes = current.nbytes
    clone = recv.__getitem__.return_value.view.return_value.reshape.return_value.clone
    clone.side_effect = RuntimeError("clone failed")
    operations = []
    stream = MagicMock()
    stream.synchronize.side_effect = lambda: operations.append("ready")
    store._backend.batch_get_into_buffers.side_effect = lambda *args: operations.append("get") or [0]
    store._backend.unregister_buffer.side_effect = lambda *args: operations.append("unregister")
    npu = SimpleNamespace(current_stream=MagicMock(return_value=stream))
    try:
        with (
            patch.object(store_module.torch, "npu", npu, create=True),
            patch.object(store_module.torch, "empty", return_value=recv),
            pytest.raises(RuntimeError, match="clone failed"),
        ):
            store.select_ssim_from_oversized_batches(
                [[ResizedTensorLoadRequest(0, "candidate", current.nbytes)]], current, 0.99
            )
        assert operations == ["ready", "get", "ready", "unregister"]
    finally:
        store.shutdown()


def test_device_ssim_scoring_stays_on_caller_thread():
    store = _store()
    # Meta tensors check dispatch without running accelerator arithmetic.
    current = torch.ones(16, 12, device="meta")
    count = store_module._PARALLEL_SIMILARITY_COMPUTE_THRESHOLD + 1
    requests = [
        ResizedTensorLoadRequest(
            i,
            store.get_resized_key(f"candidate-{i}", (1, 4, 4), current.shape, current.dtype),
            current.nbytes,
        )
        for i in range(count)
    ]
    caller = get_ident()
    worker_threads = []
    try:
        with patch.object(store, "_score_candidate", side_effect=lambda *args: worker_threads.append(get_ident())):
            assert store.calculate_ssim_results(current, [current] * count, requests) == []
        assert worker_threads == [caller] * count
    finally:
        store.shutdown()
