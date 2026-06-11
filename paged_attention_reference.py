import math
import torch 

from kv_cache_pool import KVCachePool


def gather_kv_for_sequence(
        cache_pool: KVCachePool,
        layer_id: int,
        block_table: list[int],
        seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gathers a request's logical K/V sequence from physical paged KV blocks.

    Returns:
        keys: [seq_len, num_kv_heads, head_dim]
        values: [seq_len, num_kv_heads, head_dim]
    """

    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative: {seq_len}")
    
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []

    for token_position in range(seq_len):
        key, value = cache_pool.read_request_token(
            layer_id=layer_id,
            block_table=block_table,
            token_position=token_position
        )

        keys.append(key)
        values.append(value)
    
    if seq_len == 0:
        empty_shape = (
            0,
            cache_pool.layout.num_kv_heads,
            cache_pool.layout.head_dim
        )

        return (
            torch.empty(
                empty_shape,
                device=cache_pool.key_cache.device,
                dtype = cache_pool.key_cache.dtype
            ),
            torch.empty(
                empty_shape,
                device=cache_pool.value_cache.device,
                dtype= cache_pool.value_cache.dtype
            )
        )
    
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


def repeat_kv_heads_for_query_heads(
        tensor: torch.Tensor,
        num_query_heads:int,
) -> torch.Tensor:
    """
    Expands K/V heads to match query heads for MHA/GQA/MQA.

    Input:
        tensor: [seq_len, num_kv_heads, head_dim]
    
    Output:
        tensor: [seq_len, num_query_heads, head_dim]
    
    Constraint:
        num_query_heads must be divisible by num_kv_heads.
    """

    seq_len, num_kv_heads, head_dim = tensor.shape

    if num_query_heads == num_kv_heads:
        return tensor
    
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    
    repeats = num_query_heads // num_kv_heads
    return tensor.repeat_interleave(repeats,dim=1)


def paged_attention_decode_reference(
        q: torch.Tensor,
        cache_pool: KVCachePool,
        layer_id: int,
        block_table: list[int],
        seq_len: int,
        scale: float | None = None
) -> torch.Tensor:
    """
    Reference decode attention for one sequence.

    Args:
        q:
            [num_query_heads, head_dim]
        
        cache_pool:
            Tensor-backed paged KV cache.
        
        layer_id:
            Which transformer layer's KV cache to read
        
        block_table:
            Request-local logical block -> physical block mapping
        
        seq_len:
            Number of K/V tokens visible to attention.
        
        scale:
            Optional attention scale. Defaults to 1 / sqrt(head_dim)
    
    Returns:
        output:
            [num_query_heads, head_dim]
    """
    if q.ndim != 2:
        raise ValueError(f"q must have shape [num_query_heads, head_dim] got {q.shape}")
    
    num_query_heads, head_dim = q.shape

    if head_dim != cache_pool.layout.head_dim:
        raise ValueError(
            f"q head_dim={head_dim} does not match layout head_dim="
            f"{cache_pool.layout.head_dim}"
        )
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    keys, values = gather_kv_for_sequence(
        cache_pool=cache_pool,
        layer_id=layer_id,
        block_table=block_table,
        seq_len=seq_len
    )

    keys = repeat_kv_heads_for_query_heads(
        tensor=keys,
        num_query_heads=num_query_heads
    )

    values = repeat_kv_heads_for_query_heads(
        tensor=values,
        num_query_heads=num_query_heads
    )


    #q: [num_query_heads, head_dim]
    #keys: [seq_len, num_query_heads, head_dim]
    #scores: [num_query_heads, seq_len]
    scores = torch.einsum("hd,shd->hs",q.float(),keys.float()) * scale

    probs = torch.softmax(scores, dim=-1)

    #probs: [num_query_heads, seq_len]
    #values: [seq_len, num_query_heads, head_dim]
    #output: [num_query_heads, head_dim]

    output = torch.einsum("hs,shd->hd",probs,values.float())

    return output.to(dtype=q.dtype)


def paged_attention_decode_batch_reference(
    q: torch.Tensor,
    cache_pool: KVCachePool,
    layer_id: int,
    block_tables: list[list[int]],
    seq_lens: list[int],
    scale: float | None = None,
) -> torch.Tensor:
    """
    Reference decode attention for a batch of sequences.

    Args:
        q:
            [batch_size, num_query_heads, head_dim]

        block_tables:
            list of block tables, length batch_size

        seq_lens:
            list of sequence lengths, length batch_size

    Returns:
        output:
            [batch_size, num_query_heads, head_dim]
    """

    if q.ndim != 3:
        raise ValueError(
            f"q must have shape [batch_size, num_query_heads, head_dim], got {q.shape}"
        )

    batch_size = q.shape[0]

    if len(block_tables) != batch_size:
        raise ValueError(
            f"len(block_tables)={len(block_tables)} does not match "
            f"batch_size={batch_size}"
        )

    if len(seq_lens) != batch_size:
        raise ValueError(
            f"len(seq_lens)={len(seq_lens)} does not match batch_size={batch_size}"
        )

    outputs = []

    for batch_index in range(batch_size):
        output = paged_attention_decode_reference(
            q=q[batch_index],
            cache_pool=cache_pool,
            layer_id=layer_id,
            block_table=block_tables[batch_index],
            seq_len=seq_lens[batch_index],
            scale=scale,
        )
        outputs.append(output)

    return torch.stack(outputs, dim=0)
     