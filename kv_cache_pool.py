import torch

from kv_cache_layout import KVCacheLayout


class KVCachePool:
    """
    Tensor-backed physical KV cache pool.

    This owns the physical key/value tensors for a paged KV layout.

    Conceptual index:

        key_cache[layer_id, physical_block_id, block_offset, kv_head, head_dim]
        value_cache[layer_id, physical_block_id, block_offset, kv_head, head_dim]
    
    For now this is not wired to the model attention. It validates memory layout
    and provides write/read helpers that mirror the future CUDA backend
    """

    def __init__(self, layout: KVCacheLayout) -> None:
        self.layout = layout

        self.key_cache = torch.empty(
            layout.key_cache_shape,
            device=layout.device,
            dtype= layout.torch_dtype,
        )

        self.value_cache = torch.empty(
            layout.value_cache_shape,
            device=layout.device,
            dtype=layout.torch_dtype
        )
    

    def zero_(self) -> None:
        self.key_cache.zero_()
        self.value_cache.zero_()
    

    def write_token(
            self,
            layer_id: int,
            physical_block_id: int,
            block_offset: int,
            key: torch.Tensor,
            value: torch.Tensor
    ) -> None:
        """
        writes one token's K/V vectors for one layer into the physical cache.

        Expected key/value shape:

            [num_kv_heads, head_dim]
        """

        self._validate_layer_id(layer_id)
        self._validate_block_address(physical_block_id, block_offset)
        self._validate_kv_tensor(key, "key")
        self._validate_kv_tensor(value,"value")

        self.key_cache[layer_id, physical_block_id, block_offset].copy_(key)
        self.value_cache[layer_id, physical_block_id, block_offset].copy_(value)

    
    def read_token(
            self,
            layer_id: int,
            physical_block_id: int,
            block_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reads one token's K/V vectors for one layer.

        Returned tensors have shape:

            [num_kv_heads, head_dim] 
        """

        self._validate_layer_id(layer_id)
        self._validate_block_address(physical_block_id, block_offset)

        key = self.key_cache[layer_id, physical_block_id, block_offset]
        value = self.value_cache[layer_id, physical_block_id, block_offset]

        return key, value
    
    def write_request_token(
            self,
            layer_id: int,
            block_table: list[int],
            token_position: int,
            key: torch.Tensor,
            value: torch.Tensor
    ) -> tuple[int,int]:
        """
        Writes a token using request-local token_position + block_table.

        Returns:
            physical_block_id, block_offset
        """

        physical_block_id, block_offset = self.layout.locate_token(
            block_table=block_table,
            token_position=token_position
        )

        self.write_token(
            layer_id=layer_id,
            physical_block_id=physical_block_id,
            block_offset=block_offset,
            key=key,
            value=value
        )
        return physical_block_id, block_offset
    
    
    def read_request_token(
            self,
            layer_id: int,
            block_table: list[int],
            token_position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reads a token using request-local token_position + block_table.
        """

        physical_block_id, block_offset = self.layout.locate_token(
            block_table=block_table,
            token_position=token_position
        )

        return self.read_token(
            layer_id=layer_id,
            physical_block_id=physical_block_id,
            block_offset=block_offset
        )
    

    def bytes_per_cache(self) -> int:
        return self.key_cache.numel() * self.key_cache.element_size()
    
    def total_bytes(self) -> int:
        return self.bytes_per_cache() + (
            self.value_cache.numel() * self.value_cache.element_size()
        )
    
    def snapshot(self) -> dict:
        return {
            "layout": self.layout.snapshot(),
            "key_cache_shape": tuple(self.key_cache.shape),
            "value_cache_shape": tuple(self.value_cache.shape),
            "dtype": str(self.key_cache.dtype),
            "device": str(self.key_cache.device),
            "key_cache_bytes": self.bytes_per_cache(),
            "value_cache_bytes": self.value_cache.numel()
            * self.value_cache.element_size(),
            "total_bytes": self.total_bytes(),
            "total_mib": self.total_bytes() / (1024 * 1024),
        }

    def _validate_layer_id(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.layout.num_layers:
            raise IndexError(
                f"layer_id={layer_id} out of range for "
                f"num_layers={self.layout.num_layers}"
            )

    def _validate_block_address(
        self,
        physical_block_id: int,
        block_offset: int,
    ) -> None:
        if physical_block_id < 0 or physical_block_id >= self.layout.total_blocks:
            raise IndexError(
                f"physical_block_id={physical_block_id} out of range for "
                f"total_blocks={self.layout.total_blocks}"
            )

        if block_offset < 0 or block_offset >= self.layout.block_size_tokens:
            raise IndexError(
                f"block_offset={block_offset} out of range for "
                f"block_size_tokens={self.layout.block_size_tokens}"
            )

    def _validate_kv_tensor(self, tensor: torch.Tensor, name: str) -> None:
        expected_shape = (
            self.layout.num_kv_heads,
            self.layout.head_dim,
        )

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} tensor has shape={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )

        if tensor.device != self.key_cache.device:
            raise ValueError(
                f"{name} tensor device={tensor.device}, "
                f"expected={self.key_cache.device}"
            )

        if tensor.dtype != self.key_cache.dtype:
            raise ValueError(
                f"{name} tensor dtype={tensor.dtype}, "
                f"expected={self.key_cache.dtype}"
            )