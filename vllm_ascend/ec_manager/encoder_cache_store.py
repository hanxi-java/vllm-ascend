from abc import ABC, abstractmethod
import threading
from typing import ClassVar

import torch


class EncoderCacheStore(ABC):
    """Abstract interface for storing encoder output tensors.

    Implementations may store tensors in local CPU memory, shared CPU memory,
    memcache, Mooncake, or other external stores.
    """

    @abstractmethod
    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        """Store ``tensor`` under ``mm_hash``."""
        ...

    @abstractmethod
    def get(self, mm_hash: str) -> torch.Tensor | None:
        """Return the tensor for ``mm_hash`` if present, else ``None``."""
        ...

    @abstractmethod
    def pop(self, mm_hash: str) -> torch.Tensor | None:
        """Remove and return the tensor for ``mm_hash`` if present."""
        ...

    @abstractmethod
    def contains(self, mm_hash: str) -> bool:
        """Return whether a tensor is stored for ``mm_hash``."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored tensors."""
        ...

    def __contains__(self, mm_hash: str) -> bool:
        return self.contains(mm_hash)


class CPUEncoderCacheStore(EncoderCacheStore):
    """In-process CPU tensor store backed by a plain dict.

    This store is a process-local singleton so that all consumers in the same
    worker process share the same CPU encoder cache.
    """

    _instance: ClassVar["CPUEncoderCacheStore | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "CPUEncoderCacheStore":
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking to ensure thread-safe singleton
                # creation while avoiding the lock on every access.
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # __init__ runs on every CPUEncoderCacheStore() call, so guard the
        # actual initialization to avoid resetting the shared cache.
        if hasattr(self, "_initialized"):
            return
        self._cache: dict[str, torch.Tensor] = {}
        self._initialized = True

    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        self._cache[mm_hash] = tensor

    def get(self, mm_hash: str) -> torch.Tensor | None:
        return self._cache.get(mm_hash)

    def pop(self, mm_hash: str) -> torch.Tensor | None:
        return self._cache.pop(mm_hash, None)

    def contains(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    def clear(self) -> None:
        self._cache.clear()
