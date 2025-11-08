"""
Generate PyTorch reference outputs for multiple batch sizes and N_latent sizes
Stores Q, K, V matrices and reference outputs in subdirectories
"""

import torch
import numpy as np
import os

N_input = 784
D = 768
DATA_DIR = 'data'
BATCHES_DIR = 'batches'
SEED = 42

BATCH_SIZES = [4, 8, 16, 32]

N_latent_values = [64, 128, 256, 512, 1024]

def compute_cross_attention(Q, K, V, D):
    """Compute cross-attention: softmax(Q @ K^T / sqrt(d)) @ V"""
    # Q: [BATCH_SIZE, N_latent, D], K: [BATCH_SIZE, N_input, D], V: [BATCH_SIZE, N_input, D]
    scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(D)  # [BATCH_SIZE, N_latent, N_input]
    attn_weights = torch.softmax(scores, dim=-1)  # [BATCH_SIZE, N_latent, N_input]
    output = torch.bmm(attn_weights, V)  # [BATCH_SIZE, N_latent, D]
    return output

def generate_reference_for_batch_size(BATCH_SIZE, N_latent, N_input, D, seed=42):
    """Generate Q, K, V matrices and reference output for a given batch size and N_latent."""
    np.random.seed(seed)

    Q = np.random.randn(BATCH_SIZE, N_latent, D).astype(np.float32)
    K = np.random.randn(BATCH_SIZE, N_input, D).astype(np.float32)
    V = np.random.randn(BATCH_SIZE, N_input, D).astype(np.float32)

    Q_torch = torch.from_numpy(Q)  # [BATCH_SIZE, N_latent, D]
    K_torch = torch.from_numpy(K)  # [BATCH_SIZE, N_input, D]
    V_torch = torch.from_numpy(V)  # [BATCH_SIZE, N_input, D]

    output_torch = compute_cross_attention(Q_torch, K_torch, V_torch, D)

    output_np = output_torch.detach().cpu().numpy().astype(np.float32)  # [BATCH_SIZE, N_latent, D]

    return Q, K, V, output_np

def main():
    """Generate reference outputs for all batch sizes and N_latent sizes."""
    print("="*70)
    print("Generating PyTorch Reference Outputs for Multiple Batch Sizes")
    print("="*70)
    print(f"Fixed dimensions:")
    print(f"  N_input:  {N_input}")
    print(f"  D:         {D}")
    print(f"\nBatch sizes: {BATCH_SIZES}")
    print(f"N_latent values: {N_latent_values}")
    print("="*70)

    os.makedirs(DATA_DIR, exist_ok=True)
    batches_dir = os.path.join(DATA_DIR, BATCHES_DIR)
    os.makedirs(batches_dir, exist_ok=True)

    for BATCH_SIZE in BATCH_SIZES:
        print(f"\n{'='*70}")
        print(f"Processing Batch Size = {BATCH_SIZE}")
        print(f"{'='*70}")

        batch_dir = os.path.join(batches_dir, f'batch_{BATCH_SIZE}')
        os.makedirs(batch_dir, exist_ok=True)

        for N_latent in N_latent_values:
            print(f"\n  N_latent = {N_latent}...")

            subdir = os.path.join(batch_dir, f'N_latent_{N_latent}')
            os.makedirs(subdir, exist_ok=True)

            Q, K, V, output_ref = generate_reference_for_batch_size(BATCH_SIZE, N_latent, N_input, D, SEED)

            np.save(os.path.join(subdir, 'Q_matrix.npy'), Q)
            np.save(os.path.join(subdir, 'K_matrix.npy'), K)
            np.save(os.path.join(subdir, 'V_matrix.npy'), V)
            np.save(os.path.join(subdir, 'output_reference.npy'), output_ref)

            print(f"    Saved to {subdir}/")
            print(f"      Q: {Q.shape}, K: {K.shape}, V: {V.shape}, Output: {output_ref.shape}")

    print("\n" + "="*70)
    print("Reference generation complete!")
    print("="*70)
    print(f"\nDirectory structure:")
    print(f"  {DATA_DIR}/")
    print(f"    {BATCHES_DIR}/")
    for BATCH_SIZE in BATCH_SIZES:
        print(f"      batch_{BATCH_SIZE}/")
        for N_latent in N_latent_values:
            print(f"        N_latent_{N_latent}/")
            print(f"          Q_matrix.npy")
            print(f"          K_matrix.npy")
            print(f"          V_matrix.npy")
            print(f"          output_reference.npy")
    print("="*70)

if __name__ == "__main__":
    main()
