from dataclasses import dataclass

import torch

from kv_block_manager import KVBlockManager
from request_state import RequestState


@dataclass(frozen=True)
class DecodeBatch:
    """
    Kernel-shaped metadata for one decode iteration.

    This is not the final CUDA object, but it intentionally mirrors the
    data a CUDA C++ paged-attention backend will eventually consume.
    """

    request_ids: list[str]
    input_token_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)
    

    @property
    def max_blocks_per_seq(self) -> int:
        # .numel() returns number of elements in tensor in O(1) time
        if self.block_tables.numel() == 0:
            return 0
        return self.block_tables.shape[1]
    
    def snapshot(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "request_ids": self.request_ids,
            "input_token_ids_shape": tuple(self.input_token_ids.shape),
            "positions": self.positions.detach().cpu().tolist(),
            "seq_lens": self.seq_lens.detach().cpu().tolist(),
            "block_tables_shape": tuple(self.block_tables.shape),
            "block_tables": self.block_tables.detach().cpu().tolist(),
        }
    

def build_decode_batch(
            request_states: list[RequestState],
            kv_block_manager: KVBlockManager,
            device: str,
    ) -> DecodeBatch:
        
        active_requests = [
            request_state
            for request_state in request_states
            if request_state.status != "finished"
        ]

        request_ids: list[str] = []
        input_token_ids: list[int] = []
        positions: list[int] = []
        seq_lens: list[int] = []
        block_tables_list: list[list[int]] = []

        max_blocks_per_seq = 0

        for request_state in active_requests:
            if request_state.next_token is None:
                raise ValueError(
                    f"request_id={request_state.request_id} has no next_token"
                )
            
            request_id = str(request_state.request_id)
            block_table = kv_block_manager.get_block_tables(request_id)

            # The next token will be written at the current sequence length.
            position = request_state.prompt_tokens + request_state.generated_tokens
            seq_len = position + 1

            request_ids.append(request_id)
            input_token_ids.append(int(request_state.next_token.item()))
            positions.append(position)
            seq_lens.append(seq_len)
            block_tables_list.append(block_table)

            max_blocks_per_seq = max(max_blocks_per_seq, len(block_table))
        
        padded_block_tables: list[list[int]] = []

        for block_table in block_tables_list:
            padded = block_table + [-1] * (max_blocks_per_seq - len(block_table))
            padded_block_tables.append(padded)

        if request_ids:
            input_token_ids_tensor = torch.tensor(
                input_token_ids,
                dtype=torch.int32,
                device=device
            )

            positions_tensor = torch.tensor(
                positions,
                dtype=torch.int32,
                device=device,
            )

            seq_lens_tensor = torch.tensor(
                seq_lens,
                dtype=torch.int32,
                device=device
            )

            block_tables_tensor = torch.tensor(
                padded_block_tables,
                dtype=torch.int32,
                device=device
            )
        
        else:
            input_token_ids_tensor = torch.empty((0,), dtype=torch.int32, device=device)
            positions_tensor = torch.empty((0,), dtype=torch.int32, device=device)
            seq_lens_tensor = torch.empty((0,), dtype=torch.int32, device=device)
            block_tables_tensor = torch.empty((0, 0), dtype=torch.int32, device=device)
        
        return DecodeBatch(
            request_ids=request_ids,
            input_token_ids=input_token_ids_tensor,
            positions=positions_tensor,
            seq_lens=seq_lens_tensor,
            block_tables=block_tables_tensor,
        )

