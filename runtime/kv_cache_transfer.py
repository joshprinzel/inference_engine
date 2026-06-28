from __future__ import annotations

import torch

from runtime.kv_cache_pool import KVCachePool


PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = list[PastKeyValue]


def write_past_key_values_to_pool(
    kv_cache_pool: KVCachePool,
    block_table: list[int],
    past_key_values: PastKeyValues,
    start_token_position: int = 0,
) -> None:
    """
    Write contiguous per-layer K/V tensors into KVCachePool.

    Input K/V shape per layer:
        key:   [batch, kv_heads, seq_len, head_dim]
        value: [batch, kv_heads, seq_len, head_dim]

    KVCachePool token write shape:
        key[token]:   [kv_heads, head_dim]
        value[token]: [kv_heads, head_dim]

    This helper currently assumes batch_size == 1 because RequestState represents
    one logical request.
    """

    for layer_id, (layer_key, layer_value) in enumerate(past_key_values):
        batch_size, _, seq_len, _ = layer_key.shape

        if batch_size != 1:
            raise ValueError(
                f"KVCachePool writeback currently expects batch_size=1, "
                f"got batch_size={batch_size}"
            )

        if layer_value.shape != layer_key.shape:
            raise ValueError(
                f"Layer {layer_id} value shape={tuple(layer_value.shape)} does not "
                f"match key shape={tuple(layer_key.shape)}"
            )

        for local_index in range(seq_len):
            token_position = start_token_position + local_index

            token_key = layer_key[0, :, local_index, :].contiguous()
            token_value = layer_value[0, :, local_index, :].contiguous()

            kv_cache_pool.write_request_token(
                layer_id=layer_id,
                block_table=block_table,
                token_position=token_position,
                key=token_key,
                value=token_value,
            )


def write_last_token_past_key_values_to_pool(
    kv_cache_pool: KVCachePool,
    block_table: list[int],
    past_key_values: PastKeyValues,
    token_position: int,
) -> None:
    """
    Write only the final token from each layer's present K/V tensors.

    Input K/V shape per layer:
        key:   [1, kv_heads, seq_len, head_dim]
        value: [1, kv_heads, seq_len, head_dim]

    This is useful after one-token cached decode, where present_key_values contain
    the full cache but only the final position is newly produced.
    """

    for layer_id, (layer_key, layer_value) in enumerate(past_key_values):
        batch_size, _, seq_len, _ = layer_key.shape

        if batch_size != 1:
            raise ValueError(
                f"KVCachePool writeback currently expects batch_size=1, "
                f"got batch_size={batch_size}"
            )

        if token_position < 0 or token_position >= seq_len:
            raise IndexError(
                f"token_position={token_position} out of range for "
                f"present seq_len={seq_len}"
            )

        token_key = layer_key[0, :, token_position, :].contiguous()
        token_value = layer_value[0, :, token_position, :].contiguous()

        kv_cache_pool.write_request_token(
            layer_id=layer_id,
            block_table=block_table,
            token_position=token_position,
            key=token_key,
            value=token_value,
        )


def gather_past_key_values_from_pool(
    kv_cache_pool: KVCachePool,
    block_table: list[int],
    seq_len: int,
) -> PastKeyValues:
    """
    Gather request-local K/V from KVCachePool back into contiguous format.

    Output K/V shape per layer:
        key:   [1, kv_heads, seq_len, head_dim]
        value: [1, kv_heads, seq_len, head_dim]
    """

    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")

    gathered: PastKeyValues = []
    layout = kv_cache_pool.layout

    for layer_id in range(layout.num_layers):
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        for token_position in range(seq_len):
            token_key, token_value = kv_cache_pool.read_request_token(
                layer_id=layer_id,
                block_table=block_table,
                token_position=token_position,
            )

            keys.append(token_key)
            values.append(token_value)

        if seq_len == 0:
            layer_key = torch.empty(
                1,
                layout.num_kv_heads,
                0,
                layout.head_dim,
                device=kv_cache_pool.key_cache.device,
                dtype=kv_cache_pool.key_cache.dtype,
            )
            layer_value = torch.empty_like(layer_key)
        else:
            # [seq_len, kv_heads, head_dim]
            layer_key = torch.stack(keys, dim=0)
            layer_value = torch.stack(values, dim=0)

            # [kv_heads, seq_len, head_dim]
            layer_key = layer_key.transpose(0, 1).contiguous()
            layer_value = layer_value.transpose(0, 1).contiguous()

            # [1, kv_heads, seq_len, head_dim]
            layer_key = layer_key.unsqueeze(0)
            layer_value = layer_value.unsqueeze(0)

        gathered.append((layer_key, layer_value))

    return gathered