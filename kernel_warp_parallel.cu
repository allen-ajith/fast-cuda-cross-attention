/**
 * Block-Parallel CUDA Kernel for Perceiver Cross-Attention
 * Warp-level cooperation: each warp processes one latent vector
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define N_LATENT 512
#define N_INPUT 784
#define D 768

__global__ void parallel_cross_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int batch_size,
    int N_latent,
    int N_input,
    int dim
) {
    __shared__ float shared_scores[4][784];
    __shared__ float shared_sum[4];

    int batch_idx = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int latent_idx = blockIdx.x * 4 + warp_id;

    if (batch_idx >= batch_size || latent_idx >= N_latent) return;

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

    float max_score = -FLT_MAX;
    for (int k = 0; k < N_input; k++) {
        float partial_dot = 0.0f;

        for (int i = 0; i < elems_per_thread; i++) {
            int idx = lane_id + i * 32;
            if (idx < dim) {
                partial_dot += my_q[i] * K[K_batch_offset + k * dim + idx];
            }
        }

        unsigned mask = __activemask();
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_dot += __shfl_down_sync(mask, partial_dot, offset);
        }

        if (lane_id == 0) {
            float score = partial_dot / sqrtf((float)dim);
            shared_scores[warp_id][k] = score;
            max_score = fmaxf(max_score, score);
        }
    }

    if (lane_id == 0) {
        float sum_exp = 0.0f;
        for (int k = 0; k < N_input; k++) {
            float exp_score = expf(shared_scores[warp_id][k] - max_score);
            shared_scores[warp_id][k] = exp_score;
            sum_exp += exp_score;
        }
        shared_sum[warp_id] = sum_exp;
    }
    __syncwarp();

    float my_output[24] = {0.0f};
    float norm = 1.0f / shared_sum[warp_id];

    for (int k = 0; k < N_input; k++) {
        float weight = shared_scores[warp_id][k] * norm;
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = lane_id + i * 32;
            if (idx < dim) {
                my_output[i] += weight * V[V_batch_offset + k * dim + idx];
            }
        }
    }

    for (int i = 0; i < elems_per_thread; i++) {
        int idx = lane_id + i * 32;
        if (idx < dim) {
            output[out_batch_offset + latent_idx * dim + idx] = my_output[i];
        }
    }
}

extern "C" void launch_parallel_kernel(
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

    parallel_cross_attention<<<gridSize, blockSize, 0, stream>>>(
        Q, K, V, output, batch_size, N_latent, N_input, dim
    );
}
