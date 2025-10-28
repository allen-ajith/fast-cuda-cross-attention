"""
Generate PyTorch reference outputs for multiple N_latent sizes
Stores Q, K, V matrices and reference outputs in subdirectories
"""

import torch
import numpy as np
import os

N_input = 784
D = 768
DATA_DIR = 'data'
SEED = 42

N_latent_values = [64, 128, 256, 512, 1024]

def compute_cross_attention(Q, K, V, D):
    """Compute cross-attention: softmax(Q @ K^T / sqrt(d)) @ V"""
    # Q: [1, N_latent, D], K: [1, N_input, D], V: [1, N_input, D]
    scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(D)  # [1, N_latent, N_input]
    attn_weights = torch.softmax(scores, dim=-1)  # [1, N_latent, N_input]
    output = torch.bmm(attn_weights, V)  # [1, N_latent, D]
    return output

def generate_reference_for_size(N_latent, N_input, D, seed=42):
    """Generate Q, K, V matrices and reference output for a given N_latent."""
    np.random.seed(seed)

    Q = np.random.randn(N_latent, D).astype(np.float32)
    K = np.random.randn(N_input, D).astype(np.float32)
    V = np.random.randn(N_input, D).astype(np.float32)

    Q_torch = torch.from_numpy(Q).unsqueeze(0)  # [1, N_latent, D]
    K_torch = torch.from_numpy(K).unsqueeze(0)  # [1, N_input, D]
    V_torch = torch.from_numpy(V).unsqueeze(0)  # [1, N_input, D]

    output_torch = compute_cross_attention(Q_torch, K_torch, V_torch, D)

    output_np = output_torch.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [N_latent, D]
    output_np = output_np.reshape(1, N_latent, D)

    return Q, K, V, output_np

def main():
    """Generate reference outputs for all N_latent sizes."""
    print("="*70)
    print("Generating PyTorch Reference Outputs for Multiple N_latent Sizes")
    print("="*70)
    print(f"Fixed dimensions:")
    print(f"  N_input:  {N_input}")
    print(f"  D:         {D}")
    print(f"\nGenerating references for N_latent: {N_latent_values}")
    print("="*70)

    os.makedirs(DATA_DIR, exist_ok=True)

    for N_latent in N_latent_values:
        print(f"\nProcessing N_latent = {N_latent}...")

        subdir = os.path.join(DATA_DIR, f'N_latent_{N_latent}')
        os.makedirs(subdir, exist_ok=True)

        Q, K, V, output_ref = generate_reference_for_size(N_latent, N_input, D, SEED)

        np.save(os.path.join(subdir, 'Q_matrix.npy'), Q)
        np.save(os.path.join(subdir, 'K_matrix.npy'), K)
        np.save(os.path.join(subdir, 'V_matrix.npy'), V)
        np.save(os.path.join(subdir, 'output_reference.npy'), output_ref)

        print(f"  Saved to {subdir}/")
        print(f"    Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        print(f"    Output reference: {output_ref.shape}")

    print("\n" + "="*70)
    print("Reference generation complete!")
    print("="*70)
    print(f"\nDirectory structure:")
    print(f"  {DATA_DIR}/")
    for N_latent in N_latent_values:
        print(f"    N_latent_{N_latent}/")
        print(f"      Q_matrix.npy")
        print(f"      K_matrix.npy")
        print(f"      V_matrix.npy")
        print(f"      output_reference.npy")
    print("="*70)

if __name__ == "__main__":
    main()
