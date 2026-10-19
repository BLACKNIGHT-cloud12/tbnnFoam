import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def comp_inv_and_bases(grad_U, omega):
    N = grad_U.shape[0]
    tau = 1.0 / np.clip(omega, 0.0001, 1000000.0)
    S = 0.5 * (grad_U + np.swapaxes(grad_U, 1, 2))
    W = 0.5 * (grad_U - np.swapaxes(grad_U, 1, 2))
    s = S * tau[:, None, None]
    w = W * tau[:, None, None]
    s2 = np.matmul(s, s)
    w2 = np.matmul(w, w)
    s3 = np.matmul(s2, s)
    w2s = np.matmul(w2, s)
    w2s2 = np.matmul(w2, s2)
    I1 = np.trace(s2, axis1=1, axis2=2)[:, None]
    I2 = np.trace(w2, axis1=1, axis2=2)[:, None]
    I3 = np.trace(s3, axis1=1, axis2=2)[:, None]
    I4 = np.trace(w2s, axis1=1, axis2=2)[:, None]
    I5 = np.trace(w2s2, axis1=1, axis2=2)[:, None]
    invariants = np.hstack([I1, I2, I3, I4, I5])
    I = np.eye(3)[None, :, :]
    T = np.zeros((N, 10, 3, 3))
    T[:, 0] = s
    T[:, 1] = np.matmul(s, w) - np.matmul(w, s)
    T[:, 2] = s2 - 1.0 / 3.0 * I1[:, :, None] * I
    T[:, 3] = w2 - 1.0 / 3.0 * I2[:, :, None] * I
    T[:, 4] = np.matmul(w, s2) - np.matmul(s2, w)
    T[:, 5] = np.matmul(w2, s) + np.matmul(s, w2) - 2.0 / 3.0 * I4[:, :, None] * I
    T[:, 6] = np.matmul(np.matmul(w, s), w2) - np.matmul(np.matmul(w2, s), w)
    T[:, 7] = np.matmul(np.matmul(s, w), s2) - np.matmul(np.matmul(s2, w), s)
    T[:, 8] = np.matmul(w2, s2) + np.matmul(s2, w2) - 2.0 / 3.0 * I5[:, :, None] * I
    T[:, 9] = np.matmul(np.matmul(w, s2), w2) - np.matmul(np.matmul(w2, s2), w)
    return (invariants, T)

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
    csv_path = Path('data/benchmark_dns/square_duct_dns_Re1500.csv')
    if not csv_path.exists():
        alt_paths = [Path('../data/benchmark_dns/square_duct_dns_Re1500.csv'), Path('square_duct_dns_Re1500.csv')]
        for p in alt_paths:
            if p.exists():
                csv_path = p
                break
    if not csv_path.exists():
        raise FileNotFoundError(f'Cannot find dataset at {csv_path}. Please check file location.')
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} points from {csv_path}')
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
    print('[*] Starting training...')
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
    print('Training complete!')
    export_dir = Path('export_weights')
    export_dir.mkdir(parents=True, exist_ok=True)
    weights_dict = {'inv_mean': inv_mean.tolist(), 'inv_std': inv_std.tolist()}
    layer_idx = 0
    for name, param in model.mlp.named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()
    json_path = export_dir / 'weights.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)
    print(f'[+] Saved lightweight C++ weights to {json_path}')
    model.eval()
    example_inv = X_inv_tr[:1]
    example_basis = X_basis_tr[:1]
    traced_model = torch.jit.trace(model, (example_inv, example_basis))
    pt_path = export_dir / 'tbnn_model.pt'
    traced_model.save(str(pt_path))
    print(f'[+] Saved LibTorch TorchScript model to {pt_path}')
if __name__ == '__main__':
    train()
