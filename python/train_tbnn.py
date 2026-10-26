"""LEGACY EXPERIMENT - not the shipped model.

The shipped tbnnFoam network (5 -> 64 -> 64 -> 64 -> 10, unnormalized
invariants) is trained exclusively by tbnn_train.py at the repository
root. This script is an earlier 30-wide variant with normalized
invariants, weight decay and a tuple-returning forward that is
INCOMPATIBLE with the TorchScript consumer in tbnnFoam.C. All outputs
are therefore written to python/legacy_exports/ and never to
export_weights/ or the repository-root artifact locations, so running
it cannot clobber the audited shipped weights.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tbnn_tensor_math import comp_inv_and_bases


class TBNN(nn.Module):

    def __init__(self, hidden_dim=30):
        super(TBNN, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(5, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 10))

    def forward(self, invariants, basis_tensors):
        g = self.mlp(invariants)
        a_pred = torch.einsum('bn,bnij->bij', g, basis_tensors)
        return (a_pred, g)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    csv_path = _REPO_ROOT / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb4410_Retau150.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Cannot find dataset at {csv_path}. Run from a full checkout of tbnnFoam.')
    df = pd.read_csv(csv_path)
    print(f'Training DNS Benchmark: Re_b = 4410 | Re_h = 2205 | Re_tau = 150 (Loaded {len(df)} points from {csv_path})')
    N = len(df)
    grad_U = np.zeros((N, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_U[:, i, j] = df[f'grad_U_{i}{j}'].values
    omega = df['omega'].values
    a_true = np.zeros((N, 3, 3))
    a_true[:, 0, 0] = df['a_11'].values
    a_true[:, 1, 1] = df['a_22'].values
    a_true[:, 2, 2] = df['a_33'].values
    a_true[:, 0, 1] = a_true[:, 1, 0] = df['a_12'].values
    a_true[:, 0, 2] = a_true[:, 2, 0] = df['a_13'].values
    a_true[:, 1, 2] = a_true[:, 2, 1] = df['a_23'].values
    invariants, basis_tensors = comp_inv_and_bases(grad_U, omega)
    val_mask = (df['y'].values > 0.0) & (df['z'].values > 0.0)
    train_mask = ~val_mask
    print(f'Spatial split: {np.sum(train_mask)} training points, {np.sum(val_mask)} validation points')
    inv_mean = np.mean(invariants[train_mask], axis=0)
    inv_std = np.std(invariants[train_mask], axis=0) + 1e-06
    invariants_norm = (invariants - inv_mean) / inv_std
    X_inv_tr = torch.tensor(invariants_norm[train_mask], dtype=torch.float32).to(device)
    X_basis_tr = torch.tensor(basis_tensors[train_mask], dtype=torch.float32).to(device)
    Y_a_tr = torch.tensor(a_true[train_mask], dtype=torch.float32).to(device)
    X_inv_val = torch.tensor(invariants_norm[val_mask], dtype=torch.float32).to(device)
    X_basis_val = torch.tensor(basis_tensors[val_mask], dtype=torch.float32).to(device)
    Y_a_val = torch.tensor(a_true[val_mask], dtype=torch.float32).to(device)
    model = TBNN(hidden_dim=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
    criterion = nn.MSELoss()
    epochs = 600
    print('[*] Starting training (LEGACY 30-wide experiment)...')
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        a_pred, _ = model(X_inv_tr, X_basis_tr)
        loss = criterion(a_pred, Y_a_tr)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                a_val_pred, _ = model(X_inv_val, X_basis_val)
                val_loss = criterion(a_val_pred, Y_a_val)
                frob_err = torch.mean(torch.norm(a_val_pred - Y_a_val, dim=(1, 2))).item()
            print(f'Epoch [{epoch:4d}/{epochs}] | Train Loss: {loss.item():.6e} | Val Loss: {val_loss.item():.6e} | Val Frob: {frob_err:.6e}')
    held_csv = _REPO_ROOT / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb7000_Retau220.csv'
    if held_csv.exists():
        df_held = pd.read_csv(held_csv)
        N_h = len(df_held)
        grad_U_h = np.zeros((N_h, 3, 3))
        for i in range(3):
            for j in range(3):
                grad_U_h[:, i, j] = df_held[f'grad_U_{i}{j}'].values
        omega_h = df_held['omega'].values
        a_true_h = np.zeros((N_h, 3, 3))
        a_true_h[:, 0, 0] = df_held['a_11'].values
        a_true_h[:, 1, 1] = df_held['a_22'].values
        a_true_h[:, 2, 2] = df_held['a_33'].values
        a_true_h[:, 0, 1] = a_true_h[:, 1, 0] = df_held['a_12'].values
        a_true_h[:, 0, 2] = a_true_h[:, 2, 0] = df_held['a_13'].values
        a_true_h[:, 1, 2] = a_true_h[:, 2, 1] = df_held['a_23'].values
        inv_h, bases_h = comp_inv_and_bases(grad_U_h, omega_h)
        inv_h_norm = (inv_h - inv_mean) / inv_std
        X_inv_h = torch.tensor(inv_h_norm, dtype=torch.float32).to(device)
        X_basis_h = torch.tensor(bases_h, dtype=torch.float32).to(device)
        Y_a_h = torch.tensor(a_true_h, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            a_held_pred, _ = model(X_inv_h, X_basis_h)
            held_loss = criterion(a_held_pred, Y_a_h)
            held_frob = torch.mean(torch.norm(a_held_pred - Y_a_h, dim=(1, 2))).item()
        print(f'Held-Out Generalization (Re_b = 7000 | Re_h = 3500 | Re_tau = 220): Loss = {held_loss.item():.6e} | Frob = {held_frob:.6e}')
    print('Legacy training complete!')
    export_dir = _REPO_ROOT / 'python' / 'legacy_exports'
    export_dir.mkdir(parents=True, exist_ok=True)
    weights_dict = {'inv_mean': inv_mean.tolist(), 'inv_std': inv_std.tolist()}
    for name, param in model.mlp.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()
    json_path = export_dir / 'weights_legacy_30w.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)
    print(f'[+] Saved legacy C++ weights to {json_path}')
    model.eval()
    example_inv = X_inv_tr[:1]
    example_basis = X_basis_tr[:1]
    traced_model = torch.jit.trace(model, (example_inv, example_basis))
    pt_path = export_dir / 'tbnn_model_legacy_30w.pt'
    traced_model.save(str(pt_path))
    print(f'[+] Saved legacy TorchScript model to {pt_path}')
if __name__ == '__main__':
    train()
