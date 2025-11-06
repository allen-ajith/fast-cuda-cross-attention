/**
 * Fully Optimized CUDA Kernel for Perceiver Cross-Attention
 * Online softmax + Float4 vectorization + Tiled processing
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define TILE_K 16  // Reduced to fit in shared memory: 16*192*16 = 49KB per warp
#define D 768
#define D_VEC (D / 4)  // 192 float4s
#define TILE_K_VEC (TILE_K * D_VEC)  // Number of float4s in K tile

__global__ void optimized_cross_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_size,
    int N_latent,
    int N_input,
    int dim
) {
    __shared__ float4 shared_K_tile[TILE_K_VEC];

    float tile_scores[TILE_K];

    int batch_idx = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int latent_idx = blockIdx.x * 4 + warp_id;

    if (batch_idx >= batch_size || latent_idx >= N_latent || threadIdx.x >= 128) return;

    int Q_batch_offset = batch_idx * N_latent * dim;
    int K_batch_offset = batch_idx * N_input * dim;
    int V_batch_offset = batch_idx * N_input * dim;
    int out_batch_offset = batch_idx * N_latent * dim;

    int dim_vec = dim / 4;

    float4 my_q_vec[6];
    const float4* Q_vec = reinterpret_cast<const float4*>(Q);
    int q_base = (Q_batch_offset / 4) + latent_idx * dim_vec;

    for (int i = 0; i < 6; i++) {
        int vec_idx = lane_id + i * 32;
        if (vec_idx < dim_vec) {
            my_q_vec[i] = Q_vec[q_base + vec_idx];
        } else {
            my_q_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float4 output_accum[6];
    for (int i = 0; i < 6; i++) {
        output_accum[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    int num_tiles = (N_input + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_K;

        const float4* K_vec = reinterpret_cast<const float4*>(K);
        int total_threads = blockDim.x;
        int elems_per_thread = (TILE_K_VEC + total_threads - 1) / total_threads;
        int k_vec_batch_offset = K_batch_offset / 4;

        for (int load_idx = 0; load_idx < elems_per_thread; load_idx++) {
            int shared_idx = threadIdx.x + load_idx * total_threads;
            if (shared_idx < TILE_K_VEC) {
                int k_idx = shared_idx / D_VEC;
                int d_idx = shared_idx % D_VEC;
                int global_k = tile_start + k_idx;

                if (global_k < N_input && d_idx < dim_vec) {
                    shared_K_tile[shared_idx] = K_vec[k_vec_batch_offset + global_k * dim_vec + d_idx];
                } else {
                    shared_K_tile[shared_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        unsigned mask = __activemask();

        for (int k = 0; k < TILE_K; k++) {
            int global_k = tile_start + k;
            if (global_k < N_input) {
                float4 dot_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                for (int i = 0; i < 6; i++) {
                    int vec_idx = lane_id + i * 32;
                    if (vec_idx < dim_vec) {
                        int shared_idx = k * D_VEC + vec_idx;
                        float4 k_vec = shared_K_tile[shared_idx];
                        dot_vec.x += my_q_vec[i].x * k_vec.x;
                        dot_vec.y += my_q_vec[i].y * k_vec.y;
                        dot_vec.z += my_q_vec[i].z * k_vec.z;
                        dot_vec.w += my_q_vec[i].w * k_vec.w;
                    }
                }

                float dot = dot_vec.x + dot_vec.y + dot_vec.z + dot_vec.w;

                for (int offset = 16; offset > 0; offset /= 2) {
                    dot += __shfl_down_sync(mask, dot, offset);
                }

                if (lane_id == 0) {
                    float score = dot / sqrtf((float)dim);
                    tile_scores[k] = score;
                    tile_max = fmaxf(tile_max, score);
                }
            }
        }
        tile_max = __shfl_sync(mask, tile_max, 0);

        float scale = 1.0f;
        if (lane_id == 0) {
            float new_max = fmaxf(running_max, tile_max);
            scale = expf(running_max - new_max);
            running_sum *= scale;
            running_max = new_max;
        }
        scale = __shfl_sync(mask, scale, 0);
        running_max = __shfl_sync(mask, running_max, 0);
        running_sum = __shfl_sync(mask, running_sum, 0);
        for (int i = 0; i < 6; i++) {
            output_accum[i].x *= scale;
            output_accum[i].y *= scale;
            output_accum[i].z *= scale;
            output_accum[i].w *= scale;
        }

        float exp_scores[TILE_K];
        float tile_sum_exp = 0.0f;
        if (lane_id == 0) {
            for (int k = 0; k < TILE_K; k++) {
                int global_k = tile_start + k;
                if (global_k < N_input) {
                    exp_scores[k] = expf(tile_scores[k] - running_max);
                    tile_sum_exp += exp_scores[k];
                } else {
                    exp_scores[k] = 0.0f;
                }
            }
            running_sum += tile_sum_exp;
        }
        running_sum = __shfl_sync(mask, running_sum, 0);

        const float4* V_vec = reinterpret_cast<const float4*>(V);
        int v_vec_batch_offset = V_batch_offset / 4;
        for (int k = 0; k < TILE_K; k++) {
            int global_k = tile_start + k;
            if (global_k < N_input) {
                float exp_score = (lane_id == 0) ? exp_scores[k] : 0.0f;
                exp_score = __shfl_sync(mask, exp_score, 0);

                for (int i = 0; i < 6; i++) {
                    int vec_idx = lane_id + i * 32;
                    if (vec_idx < dim_vec) {
                        float4 v_vec = V_vec[v_vec_batch_offset + global_k * dim_vec + vec_idx];
                        output_accum[i].x += exp_score * v_vec.x;
                        output_accum[i].y += exp_score * v_vec.y;
                        output_accum[i].z += exp_score * v_vec.z;
                        output_accum[i].w += exp_score * v_vec.w;
                    }
                }
            }
        }
    }

    float norm = 1.0f / (running_sum + 1e-6f);
    float4* output_vec = reinterpret_cast<float4*>(output);
    int out_base = (out_batch_offset / 4) + latent_idx * dim_vec;

    for (int i = 0; i < 6; i++) {
        int vec_idx = lane_id + i * 32;
        if (vec_idx < dim_vec) {
            float4 out_val;
            out_val.x = output_accum[i].x * norm;
            out_val.y = output_accum[i].y * norm;
            out_val.z = output_accum[i].z * norm;
            out_val.w = output_accum[i].w * norm;
            output_vec[out_base + vec_idx] = out_val;
        }
    }
}

extern "C" void launch_optimized_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_size,
    int N_latent,
    int N_input,
    int dim,
    cudaStream_t stream
) {
    dim3 blockSize(128, 1);
    dim3 gridSize((N_latent + 3) / 4, batch_size);

    optimized_cross_attention<<<gridSize, blockSize, 0, stream>>>(
        Q, K, V, output, batch_size, N_latent, N_input, dim
    );
}
