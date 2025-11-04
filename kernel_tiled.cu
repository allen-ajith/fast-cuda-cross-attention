/**
 * Tiled CUDA Kernel with online softmax
 * Tiled processing without vectorization
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define TILE_K 16  // Tile size for shared memory caching
#define D 768
#define WARP_SIZE 32

__global__ void tiled_cross_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_size,
    int N_latent,
    int N_input,
    int dim
) {
    int batch_idx = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int latent_idx = blockIdx.x * 4 + warp_id;

    if (batch_idx >= batch_size || latent_idx >= N_latent || threadIdx.x >= 128) return;

    int Q_batch_offset = batch_idx * N_latent * dim;
    int K_batch_offset = batch_idx * N_input * dim;
    int V_batch_offset = batch_idx * N_input * dim;
    int out_batch_offset = batch_idx * N_latent * dim;

    const int elems_per_thread = dim / 32;
    float my_q[24];
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = lane_id + i * 32;
        if (idx < dim) {
            my_q[i] = Q[Q_batch_offset + latent_idx * dim + idx];
        } else {
            my_q[i] = 0.0f;
        }
    }

    float my_output[24];
    for (int i = 0; i < elems_per_thread; i++) {
        my_output[i] = 0.0f;
    }
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    __shared__ float shared_K_tile[TILE_K * D];

    float tile_scores[TILE_K];

    int num_tiles = (N_input + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_K;

        int total_threads = blockDim.x;
        int elems_per_thread_load = (TILE_K * dim + total_threads - 1) / total_threads;

        for (int load_idx = 0; load_idx < elems_per_thread_load; load_idx++) {
            int shared_idx = threadIdx.x + load_idx * total_threads;
            if (shared_idx < TILE_K * dim) {
                int k_idx = shared_idx / dim;
                int d_idx = shared_idx % dim;
                int global_k = tile_start + k_idx;

                if (global_k < N_input) {
                    shared_K_tile[shared_idx] = K[K_batch_offset + global_k * dim + d_idx];
                } else {
                    shared_K_tile[shared_idx] = 0.0f;
                }
            }
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        unsigned mask = __activemask();

        for (int k = 0; k < TILE_K; k++) {
            int global_k = tile_start + k;
            if (global_k < N_input) {
                float partial_dot = 0.0f;

                for (int i = 0; i < elems_per_thread; i++) {
                    int idx = lane_id + i * 32;
                    if (idx < dim) {
                        partial_dot += my_q[i] * shared_K_tile[k * dim + idx];
                    }
                }

                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial_dot += __shfl_down_sync(mask, partial_dot, offset);
                }

                if (lane_id == 0) {
                    float score = partial_dot / sqrtf((float)dim);
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

        for (int i = 0; i < elems_per_thread; i++) {
            my_output[i] *= scale;
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

        for (int k = 0; k < TILE_K; k++) {
            int global_k = tile_start + k;
            if (global_k < N_input) {
                float exp_score = (lane_id == 0) ? exp_scores[k] : 0.0f;
                exp_score = __shfl_sync(mask, exp_score, 0);

                for (int i = 0; i < elems_per_thread; i++) {
                    int idx = lane_id + i * 32;
                    if (idx < dim) {
                        my_output[i] += exp_score * V[V_batch_offset + global_k * dim + idx];
                    }
                }
            }
        }
    }

    float norm = 1.0f / (running_sum + 1e-6f);

    for (int i = 0; i < elems_per_thread; i++) {
        int idx = lane_id + i * 32;
        if (idx < dim) {
            output[out_batch_offset + latent_idx * dim + idx] = my_output[i] * norm;
        }
    }
}

extern "C" void launch_tiled_kernel(
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

    tiled_cross_attention<<<gridSize, blockSize, 0, stream>>>(
        Q, K, V, output, batch_size, N_latent, N_input, dim
    );
}
