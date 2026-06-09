import math
from collections import deque

class KVBlockAllocationError(RuntimeError):
    pass


class KVBlockManager:
    """
    Owns paged-KV block metadata.
    
    This does not own the actual key/value tensors yet.
    It owns the same metadata a real paged-attention backend needs.

            request_id -> block_table
    
    A block_table maps request_local logical block indices to global physical block IDs.
    """

    def __init__(
        self,
        total_blocks: int,
        block_size_tokens: int) -> None:
        if total_blocks <= 0:
            raise ValueError(f"total_blocks must be positive: {total_blocks}")
        if block_size_tokens <= 0:
            raise ValueError(f"block_size_tokens must be positive: {block_size_tokens}")
        

        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens

        self.free_blocks: deque[int] = deque(range(total_blocks))
        self.block_tables: dict[str, list[int]] = {}

    
    def blocks_needed(self, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative: {num_tokens}")
        if num_tokens == 0:
            return 0
        return math.ceil(num_tokens / self.block_size_tokens)
    
    def free_block_count(self) -> int:
        return len(self.free_blocks)
    
    def used_block_count(self) -> int:
        return self.total_blocks - self.free_block_count()
    
    def can_allocate_blocks(self, num_blocks: int) -> bool:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative: {num_blocks}")
        return self.free_block_count() >= num_blocks
    

    def can_allocate_tokens(self, num_tokens: int) -> bool:
        return self.can_allocate_blocks(self.blocks_needed(num_tokens))
    

    def allocate_blocks(
            self,
            request_id: str,
            num_blocks: int,
            
    ) -> list[int]:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be not negative: {num_blocks}")
        
        if request_id in self.block_tables:
            raise KVBlockAllocationError(
                f"request_id={request_id} already owns a block table"
            )
        
        if not self.can_allocate_blocks(num_blocks):
            raise KVBlockAllocationError(
                f"Not enough KV blocks: requested={num_blocks}, "
                f"available={self.free_block_count()}"
            )
        
        blocks = [self.free_blocks.popleft() for _ in range(num_blocks)]
        self.block_tables[request_id] = blocks

        return list(blocks)
    

    def allocate_for_tokens(
            self, 
            request_id: str,
            num_tokens: int,
    ) -> list[int]:
        return self.allocate_blocks(request_id=request_id, num_blocks=self.blocks_needed(num_tokens))
    
    def append_block(self, request_id: str) -> int:
        if request_id not in self.block_tables:
            raise KeyError(f"Unknown request_id={request_id}")
        
        if not self.free_blocks:
            raise KVBlockAllocationError("No free KV blocks available")
        
        block_id = self.free_blocks.popleft()
        self.block_tables[request_id].append(block_id)

        return block_id
    
    def ensure_capacity_for_token(
            self,
            request_id: str,
            token_position: int
    ) -> bool:
        """
        Ensures the request has enough blocks to store token_position.

        Returns true if a new block was allocated, else false.
        """

        if request_id not in self.block_tables:
            raise KeyError(f"Unknown request_id={request_id}")
        if token_position < 0:
            raise ValueError(f"token_position must be non-negative: {token_position}")
        
        required_blocks = self.blocks_needed(token_position + 1) # +1 is for eos token
        current_blocks = len(self.block_tables[request_id])

        if required_blocks <= current_blocks:
            return False
        while len(self.block_tables[request_id]) < required_blocks:
            self.append_block(request_id)

        return True
    

    def free(self, request_id: str) -> list[int]:
        blocks = self.block_tables.pop(request_id, [])

        for block_id in blocks:
            self.free_blocks.append(block_id)

        return list(blocks)
    

    def get_block_tables(self, request_id: str) -> list[int]:
        if request_id not in self.block_tables:
            raise KeyError(f"Unknown request_id={request_id}")
        
        return list(self.block_tables[request_id])
    
    def utilization(self) -> float:
        return self.used_block_count() / self.total_blocks
    

    def snapshot(self) -> dict:
        return {
            "total_blocks": self.total_blocks,
            "block_size_tokens": self.block_size_tokens,
            "used_blocks": self.used_block_count(),
            "free_blocks": self.free_block_count(),
            "utilization": self.utilization(),
            "active_requests": len(self.block_tables),
            "block_tables": {
                request_id: list(block_table)
                for request_id, block_table in self.block_tables.items()
            },
        }