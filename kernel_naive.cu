/**
 * Naive CUDA Kernel for Perceiver Cross-Attention
 * One thread per output element - no optimization
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define N_LATENT 512
#define N_INPUT 784
#define D 768

__global__ void naive_cross_attention(
    const float* Q,      // [BATCH_SIZE, N_latent, D]
    const float* K,      // [BATCH_SIZE, N_input, D]
    const float* V,      // [BATCH_SIZE, N_input, D]
    float* output,       // [BATCH_SIZE, N_latent, D]
    int batch_size,
    int N_latent,
    int N_input,
    int dim
) {
    int batch_idx = blockIdx.y;
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.z * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size || latent_idx >= N_latent || dim_idx >= dim) return;

    int Q_batch_offset = batch_idx * N_latent * dim;
    int K_batch_offset = batch_idx * N_input * dim;
    int V_batch_offset = batch_idx * N_input * dim;
    int out_batch_offset = batch_idx * N_latent * dim;

    float scores[784];

    float max_score = -FLT_MAX;
    for (int k = 0; k < N_input; k++) {
        float score = 0.0f;
        for (int d = 0; d < dim; d++) {
            score += Q[Q_batch_offset + latent_idx * dim + d] *
                     K[K_batch_offset + k * dim + d];
        }
        score /= sqrtf((float)dim);
        scores[k] = score;
        max_score = fmaxf(max_score, score);
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < N_input; k++) {
        float exp_score = expf(scores[k] - max_score);
        scores[k] = exp_score;
        sum_exp += exp_score;
    }

    float result = 0.0f;
    float norm = 1.0f / sum_exp;
    for (int k = 0; k < N_input; k++) {
        float weight = scores[k] * norm;
        result += weight * V[V_batch_offset + k * dim + dim_idx];
    }

    output[out_batch_offset + latent_idx * dim + dim_idx] = result;
}

extern "C" void launch_naive_kernel(
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
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (N_latent + blockSize.x - 1) / blockSize.x,
        batch_size,
        (dim + blockSize.y - 1) / blockSize.y
    );

    naive_cross_attention<<<gridSize, blockSize, 0, stream>>>(
        Q, K, V, output, batch_size, N_latent, N_input, dim
    );
}
