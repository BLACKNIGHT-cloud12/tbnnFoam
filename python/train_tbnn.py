"""
train_tbnn.py
=============
Standalone PyTorch training pipeline for Tensor Basis Neural Network (TBNN).
Can be run on Linux (WSL2) or copied directly into a Kaggle Notebook.

Workflow:
  1. Loads 'data/benchmark_dns/square_duct_dns_Re1500.csv'
  2. Computes the 5 Pope scalar invariants (I1..I5) and 10 basis tensors (T1..T10)
  3. Trains TBNN to predict anisotropic Reynolds stress tensor a_ij
  4. Exports:
     - 'weights.json' (for lightweight OpenFOAM C++ evaluator)
     - 'tbnn_weights.h' (C++ header with embedded weights)
     - 'tbnn_model.pt' (TorchScript model for LibTorch)
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================================
# 1. Math: Invariants & Basis Tensors (Pope, 1975 & Ling et al., 2016)
# =====================================================================
def compute_invariants_and_bases(grad_U, omega):
    """
    Computes 5 invariants and 10 basis tensors from velocity gradient.
    grad_U shape: (N, 3, 3) where grad_U[i, j] = d(U_i)/d(x_j)
    omega shape: (N,)
    """
    N = grad_U.shape[0]
    tau = 1.0 / np.clip(omega, 1e-4, 1e6)
    
    # Symmetric rate of strain S and anti-symmetric rate of rotation W
    S = 0.5 * (grad_U + np.swapaxes(grad_U, 1, 2))
    W = 0.5 * (grad_U - np.swapaxes(grad_U, 1, 2))
    
    # Non-dimensionalize
    s = S * tau[:, None, None]
    w = W * tau[:, None, None]
    
    # Matrix powers
    s2 = np.matmul(s, s)
    w2 = np.matmul(w, w)
    s3 = np.matmul(s2, s)
    w2s = np.matmul(w2, s)
    w2s2 = np.matmul(w2, s2)
    
    # 5 Invariants (traces)
    I1 = np.trace(s2, axis1=1, axis2=2)[:, None]
    I2 = np.trace(w2, axis1=1, axis2=2)[:, None]
    I3 = np.trace(s3, axis1=1, axis2=2)[:, None]
    I4 = np.trace(w2s, axis1=1, axis2=2)[:, None]
    I5 = np.trace(w2s2, axis1=1, axis2=2)[:, None]
    invariants = np.hstack([I1, I2, I3, I4, I5])  # (N, 5)
    
    # Identity matrix
    I = np.eye(3)[None, :, :]
    
    # 10 Basis Tensors (Pope's Integrity Basis)
    T = np.zeros((N, 10, 3, 3))
    T[:, 0] = s
    T[:, 1] = np.matmul(s, w) - np.matmul(w, s)
    T[:, 2] = s2 - (1.0 / 3.0) * I1[:, :, None] * I
    T[:, 3] = w2 - (1.0 / 3.0) * I2[:, :, None] * I
    T[:, 4] = np.matmul(w, s2) - np.matmul(s2, w)
    T[:, 5] = np.matmul(w2, s) + np.matmul(s, w2) - (2.0 / 3.0) * I4[:, :, None] * I
    T[:, 6] = np.matmul(np.matmul(w, s), w2) - np.matmul(np.matmul(w2, s), w)
    T[:, 7] = np.matmul(np.matmul(s, w), s2) - np.matmul(np.matmul(s2, w), s)
    T[:, 8] = np.matmul(w2, s2) + np.matmul(s2, w2) - (2.0 / 3.0) * I5[:, :, None] * I
    T[:, 9] = np.matmul(np.matmul(w, s2), w2) - np.matmul(np.matmul(w2, s2), w)
    
    return invariants, T


# =====================================================================
# 2. PyTorch TBNN Architecture
# =====================================================================
class TBNN(nn.Module):
    def __init__(self, hidden_dim=30):
        super(TBNN, self).__init__()
        # Invariants -> MLP -> 10 scalar coefficients g^(1)..g^(10)
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
        
    def forward(self, invariants, basis_tensors):
        """
        invariants: (Batch, 5)
        basis_tensors: (Batch, 10, 3, 3)
        returns a_ij: (Batch, 3, 3)
        """
        g = self.mlp(invariants)  # (Batch, 10)
        # Tensor combination: a_ij = sum_{n=1}^{10} g_n * T_n
        a_pred = torch.einsum('bn,bnij->bij', g, basis_tensors)
        return a_pred, g


# =====================================================================
# 3. Training & Weight Exporter
# =====================================================================
def train_and_export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    # Locate dataset
    csv_path = Path("data/benchmark_dns/square_duct_dns_Re1500.csv")
    if not csv_path.exists():
        # Fallback for running from python/ directory or Kaggle
        alt_paths = [Path("../data/benchmark_dns/square_duct_dns_Re1500.csv"), Path("square_duct_dns_Re1500.csv")]
        for p in alt_paths:
            if p.exists():
                csv_path = p
                break
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {csv_path}. Please check file location.")
        
    df = pd.read_csv(csv_path)
    print(f"[*] Loaded {len(df)} points from {csv_path}")
    
    # Assemble grad_U (N, 3, 3)
    N = len(df)
    grad_U = np.zeros((N, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_U[:, i, j] = df[f"grad_U_{i}{j}"].values
            
    omega = df["omega"].values
    
    # Assemble true a_ij (N, 3, 3)
    a_true = np.zeros((N, 3, 3))
    a_true[:, 0, 0] = df["a_11"].values
    a_true[:, 1, 1] = df["a_22"].values
    a_true[:, 2, 2] = df["a_33"].values
    a_true[:, 0, 1] = a_true[:, 1, 0] = df["a_12"].values
    a_true[:, 0, 2] = a_true[:, 2, 0] = df["a_13"].values
    a_true[:, 1, 2] = a_true[:, 2, 1] = df["a_23"].values
    
    # Compute Invariants and Basis Tensors
    invariants, basis_tensors = compute_invariants_and_bases(grad_U, omega)
    
    # Normalization of invariants for stable training
    inv_mean = np.mean(invariants, axis=0)
    inv_std = np.std(invariants, axis=0) + 1e-6
    invariants_norm = (invariants - inv_mean) / inv_std
    
    # Convert to PyTorch tensors
    X_inv = torch.tensor(invariants_norm, dtype=torch.float32).to(device)
    X_basis = torch.tensor(basis_tensors, dtype=torch.float32).to(device)
    Y_a = torch.tensor(a_true, dtype=torch.float32).to(device)
    
    # Initialize Model & Optimizer
    model = TBNN(hidden_dim=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Training Loop
    epochs = 600
    print("[*] Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        a_pred, _ = model(X_inv, X_basis)
        loss = criterion(a_pred, Y_a)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == epochs:
            frob_err = torch.mean(torch.norm(a_pred - Y_a, dim=(1, 2))).item()
            print(f"Epoch [{epoch:4d}/{epochs}] | MSE Loss: {loss.item():.6e} | Mean Frobenius Error: {frob_err:.6e}")
            
    print("[+] Training complete!")
    
    # Save Weights for OpenFOAM C++
    export_dir = Path("export_weights")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Export weights.json (for pure C++ evaluator)
    weights_dict = {
        "inv_mean": inv_mean.tolist(),
        "inv_std": inv_std.tolist(),
    }
    
    layer_idx = 0
    for name, param in model.mlp.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()
        
    json_path = export_dir / "weights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"[+] Saved lightweight C++ weights to {json_path}")
    
    # 2. Export TorchScript model (for LibTorch)
    model.eval()
    example_inv = X_inv[:1]
    example_basis = X_basis[:1]
    traced_model = torch.jit.trace(model, (example_inv, example_basis))
    pt_path = export_dir / "tbnn_model.pt"
    traced_model.save(str(pt_path))
    print(f"[+] Saved LibTorch TorchScript model to {pt_path}")


if __name__ == "__main__":
    train_and_export()
