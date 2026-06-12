#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace{

    constexpr int THREADS_PER_HEAD = 128;
    constexpr int MAX_SEQ_LEN = 512;

    __device__ float block_reduce_sum(float value){
        __shared__ float shared[THREADS_PER_HEAD];

        int tid = threadIdx.x;
        shared[tid] = value;
        __syncthreads();

        for(int stride = THREADS_PER_HEAD / 2; stride > 0; stride >>= 1){
            if(tid < stride){
                shared[tid] += shared[tid + stride];
            }
            __syncthreads();
        }
        return shared[0];
    }

    __global__ void paged_attention_decode_kernel_v3(
        const half* __restrict__ q,
        const half* __restrict__ key_cache,
        const half* __restrict__ value_cache,
        const int32_t* __restrict__ block_table,
        half* __restrict__ output,
        int64_t layer_id,
        int64_t seq_len,
        int64_t num_layers,
        int64_t num_blocks,
        int64_t block_size,
        int64_t num_heads,
        int64_t head_dim
    ){
        int head_id = blockIdx.x;
        int tid = threadIdx.x;

        if(head_id >= num_heads){
            return;
        }

        __shared__ float scores[MAX_SEQ_LEN];
        __shared__ float shared_max;
        __shared__ float shared_denom;

        float scale = rsqrtf(static_cast<float>(head_dim));

        /*Pass 1: 
            Compute each QK score once, store it in shared memory,
            and compute max_score for stable softmax
        */
        float max_score = -INFINITY;

        for(int64_t token_pos = 0; token_pos < seq_len; token_pos++){
            int64_t logical_block_id = token_pos / block_size;
            int64_t block_offset = token_pos % block_size;

            int64_t physical_block_id = static_cast<int64_t>(block_table[logical_block_id]);

            float partial_score = 0.0f;

            for(int64_t d = tid; d < head_dim; d+= THREADS_PER_HEAD){
                int64_t q_idx = head_id * head_dim + d;

                int64_t k_idx = (((layer_id * num_blocks + physical_block_id) * block_size + block_offset) * num_heads + head_id) * head_dim + d;

                float q_val = __half2float(q[q_idx]); //Half Precision Conversion via Nvidia Docs: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html
                float k_val = __half2float(key_cache[k_idx]);

                partial_score += q_val * k_val;
            }
            float score = block_reduce_sum(partial_score) * scale;

            if(tid == 0){
                scores[token_pos] = score;

                if(score > max_score){
                    max_score = score;
                }
            }

            // Broadcast updated max_score by making sure all threads see the same control point.
            __syncthreads();

            //All threads need the same max_score value. Since max_score is a local variable.
            if(tid == 0){
                shared_max = max_score;
            }
            __syncthreads();
            max_score = shared_max;
            __syncthreads();
        }

        //Pass 2: 
        // Compute denominator from stored scores.
        float denom = 0.0f;

        if(tid == 0){
            for(int64_t token_pos = 0; token_pos < seq_len; token_pos++){
                denom += expf(scores[token_pos] - max_score);
            }
            shared_denom = denom;
        }

        __syncthreads();
        denom = shared_denom;
        __syncthreads();

        if(!isfinite(max_score) || !isfinite(denom) || denom == 0.0f){
            for(int64_t d = tid; d < head_dim; d += THREADS_PER_HEAD){
                int64_t out_idx = head_id * head_dim + d;
                output[out_idx] = __float2half(0.0f);
            }
            return;
        }


        // Pass 3:
        // Reuse stored scores to accumulate V.
        // For this v3, head_dim <= THREADS_PER_HEAD, so each active thread owns one output dim
        int64_t output_dim = tid;
        float acc = 0.0f;

        if(output_dim < head_dim){
            for(int64_t token_pos = 0; token_pos < seq_len; token_pos++){
                int64_t logical_block_id = token_pos / block_size;
                int64_t block_offset = token_pos % block_size;
                int64_t physical_block_id = static_cast<int64_t>(block_table[logical_block_id]);

                float prob = expf(scores[token_pos] - max_score) / denom;

                int64_t v_idx = (((layer_id * num_blocks + physical_block_id) * block_size + block_offset) * num_heads + head_id) * head_dim + output_dim;
                float v_val = __half2float(value_cache[v_idx]);
                acc += prob * v_val;
            }
            int64_t out_idx = head_id * head_dim + output_dim;
            output[out_idx] = __float2half(acc);
        }
    }
} //namespace

torch::Tensor paged_attention_decode_cuda(
    torch::Tensor q,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_table,
    int64_t layer_id,
    int64_t seq_len
){
    const auto num_layers = key_cache.size(0);
    const auto num_blocks = key_cache.size(1);
    const auto block_size = key_cache.size(2);
    const auto num_heads = key_cache.size(3);
    const auto head_dim = key_cache.size(4);

    auto output = torch::empty_like(q);

    dim3 grid(num_heads);
    dim3 block(THREADS_PER_HEAD);

    paged_attention_decode_kernel_v3<<<grid,block>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        layer_id,
        seq_len,
        num_layers,
        num_blocks,
        block_size,
        num_heads,
        head_dim
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "paged_attention_decode_kernel_v2 failed: ", cudaGetErrorString(err));
    return output;
}

