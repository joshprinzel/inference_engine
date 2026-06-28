from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import sys

import torch

from .kv_cache_pool import KVCachePool
from .paged_attention_reference import paged_attention_decode_batch_reference


class AttentionBackend(ABC):
    @abstractmethod
    def decode(
        self,
        q: torch.Tensor,
        cache_pool: KVCachePool,
        layer_id: int,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


def _tensor_block_tables_to_lists(block_tables: torch.Tensor) -> list[list[int]]:
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be rank-2, got shape={tuple(block_tables.shape)}")

    block_tables_cpu = block_tables.detach().cpu().tolist()

    result: list[list[int]] = []
    for row in block_tables_cpu:
        result.append([int(block_id) for block_id in row if int(block_id) >= 0])

    return result


def _tensor_seq_lens_to_list(seq_lens: torch.Tensor) -> list[int]:
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be rank-1, got shape={tuple(seq_lens.shape)}")

    return [int(x) for x in seq_lens.detach().cpu().tolist()]


class ReferencePagedAttentionBackend(AttentionBackend):
    def decode(
        self,
        q: torch.Tensor,
        cache_pool: KVCachePool,
        layer_id: int,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        return paged_attention_decode_batch_reference(
            q=q,
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_tables=_tensor_block_tables_to_lists(block_tables),
            seq_lens=_tensor_seq_lens_to_list(seq_lens),
        )


class CudaPagedAttentionBackend:
    def __init__(self) -> None:

        repo_root = Path(__file__).resolve().parent.parent
        cuda_backend_dir = repo_root / "cuda_backend"

        if str(cuda_backend_dir) not in sys.path:
            sys.path.insert(0, str(cuda_backend_dir))

        try:
            import paged_attention_cuda
        except ImportError as exc:
            raise RuntimeError(
                "Could not import paged_attention_cuda. Build it with:\n"
                "cd cuda_backend && python setup.py build_ext --inplace"
            ) from exc

        self.op = paged_attention_cuda

    def decode(
        self,
        q: torch.Tensor,
        cache_pool: KVCachePool,
        layer_id: int,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        if q.device.type != "cuda":
            raise ValueError("CudaPagedAttentionBackend requires q on CUDA")

        if cache_pool.key_cache.device != q.device:
            raise ValueError(
                f"key_cache device={cache_pool.key_cache.device}, expected={q.device}"
            )

        if cache_pool.value_cache.device != q.device:
            raise ValueError(
                f"value_cache device={cache_pool.value_cache.device}, expected={q.device}"
            )

        if block_tables.device != q.device:
            block_tables = block_tables.to(q.device)

        if seq_lens.device != q.device:
            seq_lens = seq_lens.to(q.device)

        block_tables = block_tables.to(dtype=torch.int32)
        seq_lens = seq_lens.to(dtype=torch.int32)

        return self.op.paged_attention_decode_batch(
            q,
            cache_pool.key_cache,
            cache_pool.value_cache,
            block_tables,
            seq_lens,
            layer_id,
        )


def build_attention_backend(name: str) -> AttentionBackend:
    normalized = name.lower().strip()

    if normalized in {"reference", "ref", "python"}:
        return ReferencePagedAttentionBackend()

    if normalized in {"cuda", "cuda_paged", "paged_cuda"}:
        return CudaPagedAttentionBackend()

    raise ValueError(f"Unknown attention backend: {name}")