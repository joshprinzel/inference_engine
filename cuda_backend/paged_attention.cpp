#include <torch/extension.h>

torch::Tensor paged_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t layer_id,
    int64_t seq_len
);

torch::Tensor paged_attention_decode_batch_cuda(
    torch::Tensor q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t layer_id
);

torch::Tensor paged_attention_decode(
    torch::Tensor q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t layer_id,
    int64_t seq_len
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be CUDA");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");

    TORCH_CHECK(q.dim() == 2, "q must have shape [num_heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 5, "key_cache must have shape [num_layers, num_blocks, block_size, num_heads, head_dim]");
    TORCH_CHECK(value_cache.dim() == 5, "value_cache must have shape [num_layers, num_blocks, block_size, num_heads, head_dim]");
    TORCH_CHECK(block_table.dim() == 1, "block_table must have shape [num_blocks_for_seq]");

    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be float16 for v1");
    TORCH_CHECK(key_cache.scalar_type() == torch::kFloat16, "key_cache must be float16 for v1");
    TORCH_CHECK(value_cache.scalar_type() == torch::kFloat16, "value_cache must be float16 for v1");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");

    TORCH_CHECK(q.size(0) == key_cache.size(3), "v1 requires num_query_heads == num_kv_heads");
    TORCH_CHECK(q.size(1) == key_cache.size(4), "q head_dim must match cache head_dim");

    TORCH_CHECK(key_cache.size(0) > layer_id, "layer_id out of range");
    TORCH_CHECK(seq_len > 0, "seq_len must be positive for v1");
    TORCH_CHECK(block_table.size(0) >= (seq_len + key_cache.size(2) - 1) / key_cache.size(2),
                "block_table is too short for seq_len");

    return paged_attention_decode_cuda(
        q.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_table.contiguous(),
        layer_id,
        seq_len
    );
}

torch::Tensor paged_attention_decode_batch(
    torch::Tensor q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t layer_id
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA");
    TORCH_CHECK(value_cache.is_cuda(), "value_cache must be CUDA");
    TORCH_CHECK(block_tables.is_cuda(), "block_tables must be CUDA");
    TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA");

    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be float16");
    TORCH_CHECK(key_cache.scalar_type() == torch::kFloat16, "key_cache must be float16");
    TORCH_CHECK(value_cache.scalar_type() == torch::kFloat16, "value_cache must be float16");
    TORCH_CHECK(block_tables.scalar_type() == torch::kInt32, "block_tables must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");

    TORCH_CHECK(q.dim() == 3, "q must have shape [batch, heads, head_dim]");
    TORCH_CHECK(block_tables.dim() == 2, "block_tables must have shape [batch, max_blocks]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [batch]");

    TORCH_CHECK(key_cache.dim() == 5, "key_cache must have shape [layers, blocks, block_size, heads, head_dim]");
    TORCH_CHECK(value_cache.dim() == 5, "value_cache must have shape [layers, blocks, block_size, heads, head_dim]");

    TORCH_CHECK(q.size(0) == block_tables.size(0), "batch size mismatch");
    TORCH_CHECK(q.size(0) == seq_lens.size(0), "batch size mismatch");
    
    const auto num_query_heads = q.size(1);
    const auto num_kv_heads = key_cache.size(3);

    TORCH_CHECK(num_query_heads >= num_kv_heads, "num_query_heads must be >= num_kv_heads");
    TORCH_CHECK(num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads");
    TORCH_CHECK(q.size(2) == key_cache.size(4), "head_dim mismatch");

    TORCH_CHECK(value_cache.size(3) == key_cache.size(3), "value_cache num_kv_heads must match key_cache num_kv_heads");
    TORCH_CHECK(value_cache.size(4) == key_cache.size(4), "value_cache head_dim must match key_cache head_dim");

    return paged_attention_decode_batch_cuda(
        q.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_tables.contiguous(),
        seq_lens.contiguous(),
        layer_id
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "paged_attention_decode",
        &paged_attention_decode,
        "Paged attention decode v1"
    );

    m.def(
        "paged_attention_decode_batch",
        &paged_attention_decode_batch,
        "Paged attention decode batch"
    );
}