from dataclasses import dataclass
from typing import Literal

import torch

TorchDtypeName = Literal["float16","bfloat16","float32"]


@dataclass
class KVCacheLayout:
    """
    Describes the physical KV cache layout we want the future backend to use.

    Conceptual Layout:

        key_cache[layer_id, physical_block_id, block_offset, kv_head, head_dim]
        value_cache[layer_id, physical_block_id, block_offset, kv_head, head_dim]

    This is intentionally close to what a CUDA/TRITON paged attention backend would consume.
    """

    num_layers: int
    total_blocks: int
    block_size_tokens: int
    num_kv_heads: int
    head_dim: int 
    dtype: TorchDtypeName
    device: str

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32
        
        raise ValueError(f"Unsupported dtype: {self.dtype}")
    

    @property
    def key_cache_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_layers,
            self.total_blocks,
            self.block_size_tokens,
            self.num_kv_heads,
            self.head_dim
        )
    
    @property 
    def value_cache_shape(self) -> tuple[int,int, int,int, int]:
        return self.key_cache_shape
    

    @property
    def tokens_capacity(self) -> int:
        return self.block_size_tokens * self.total_blocks
    


    def locate_token(self, block_table: list[int], token_position: int) -> tuple[int,int]:
        """
        Maps a request-local token position to physical KV block coordinates.

        Returns:
            physical_block_id, block_offset
        """

        if token_position < 0:
            raise ValueError(f"token_position must be not negative: {token_position}")
        
        logical_block_index = token_position // self.block_size_tokens
        block_offset = token_position % self.block_size_tokens

        if logical_block_index >= len(block_table):
            raise IndexError(
                f"token_position={token_position} maps to logical_block_index="
                f"{logical_block_index}, but block_table only has "
                f"{len(block_table)} blocks"
            )
        
        physical_block_id = block_table[logical_block_index]

        if physical_block_id < 0 or physical_block_id >= self.total_blocks:
            raise IndexError(
                f"Invalid physical_block_id={physical_block_id}; "
                f"total_blocks={self.total_blocks}"
            )
        return physical_block_id, block_offset
    
    def snapshot(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "total_blocks": self.total_blocks,
            "block_size_tokens": self.block_size_tokens,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "device": self.device,
            "key_cache_shape": self.key_cache_shape,
            "value_cache_shape": self.value_cache_shape,
            "tokens_capacity": self.tokens_capacity
        }
