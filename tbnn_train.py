import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
import os
from pathlib import Path

class TensorBasisNeuralNetwork(nn.Module):

    def __init__(self, num_invariants=5, num_bases=10, hidden_dim=64):
        super(TensorBasisNeuralNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_invariants, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bases)
        )

    def forward(self, invariants: torch.Tensor, tensor_bases: torch.Tensor) -> torch.Tensor:
        invariants = torch.clamp(invariants, -50.0, 50.0)
        g = self.mlp(invariants)
        g_expanded = g.unsqueeze(-1).unsqueeze(-1)
        anisotropy = torch.sum(g_expanded * tensor_bases, dim=1)
        return anisotropy

def cardano_eigvals(A):
    p = 0.5 * np.sum(A**2)
    q = float(np.linalg.det(A))
    if p < 1e-14:
        return np.array([0.0, 0.0, 0.0])
    m = 2.0 * np.sqrt(p / 3.0)
    arg = np.clip(3.0 * q / (p * m), -1.0, 1.0)
    theta = (1.0 / 3.0) * np.arccos(arg)
    l1 = m * np.cos(theta)
    l2 = m * np.cos(theta - 2.0 * np.pi / 3.0)
    l3 = m * np.cos(theta + 2.0 * np.pi / 3.0)
    return np.sort(np.array([l1, l2, l3]))[::-1]

def project_realizable(A):
    tr = np.trace(A) / 3.0
    A_dev = A - tr * np.eye(3)
    l = cardano_eigvals(A_dev)
    scale = 1.0
    if l[2] < -1.0 / 3.0:
        scale = min(scale, (1.0 / 3.0) / abs(l[2]))
    if l[0] > 2.0 / 3.0:
        scale = min(scale, (2.0 / 3.0) / l[0])
    return scale * A_dev

def comp_inv_and_bases(grad_U, omega):
    N = grad_U.shape[0]
    tau = np.minimum(1.0 / np.clip(omega, 0.0001, 1000000.0), 5.0)
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
    inv = np.hstack([I1, I2, I3, I4, I5])
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
    return (inv, T)

def train():
    csv_file = Path('data/benchmark_dns/square_duct_dns_Reb4410_Retau150.csv')
    if not csv_file.exists():
        csv_file = Path('data/benchmark_dns/square_duct_dns_Re1500.csv')
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    N = len(data)
    print(f'Training DNS Benchmark: Re_b = 4410 | Re_h = 2205 | Re_tau = 150 (Loaded {N} points from {csv_file})')
    grad_U = np.zeros((N, 3, 3))
    omega = np.zeros(N)
    a_true = np.zeros((N, 3, 3))
    y_coords = np.zeros(N)
    z_coords = np.zeros(N)
    for idx, row in enumerate(data):
        y_coords[idx] = float(row['y'])
        z_coords[idx] = float(row['z'])
        for i in range(3):
            for j in range(3):
                grad_U[idx, i, j] = float(row[f'grad_U_{i}{j}'])
        omega[idx] = float(row['omega'])
        a_true[idx, 0, 0] = float(row['a_11'])
        a_true[idx, 1, 1] = float(row['a_22'])
        a_true[idx, 2, 2] = float(row['a_33'])
        a_true[idx, 0, 1] = a_true[idx, 1, 0] = float(row['a_12'])
        a_true[idx, 0, 2] = a_true[idx, 2, 0] = float(row['a_13'])
        a_true[idx, 1, 2] = a_true[idx, 2, 1] = float(row['a_23'])
    inv, T = comp_inv_and_bases(grad_U, omega)
    val_mask = (y_coords > 0.0) & (z_coords > 0.0)
    train_mask = ~val_mask
    print(f'Spatial split: {np.sum(train_mask)} training points, {np.sum(val_mask)} validation points')
    X_inv_train = torch.tensor(inv[train_mask], dtype=torch.float32)
    X_basis_train = torch.tensor(T[train_mask], dtype=torch.float32)
    Y_a_train = torch.tensor(a_true[train_mask], dtype=torch.float32)
    X_inv_val = torch.tensor(inv[val_mask], dtype=torch.float32)
    X_basis_val = torch.tensor(T[val_mask], dtype=torch.float32)
    Y_a_val = torch.tensor(a_true[val_mask], dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    model = TensorBasisNeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    X_inv_tr_d = X_inv_train.to(device)
    X_basis_tr_d = X_basis_train.to(device)
    Y_a_tr_d = Y_a_train.to(device)
    X_inv_val_d = X_inv_val.to(device)
    X_basis_val_d = X_basis_val.to(device)
    Y_a_val_d = Y_a_val.to(device)
    for epoch in range(1, 401):
        model.train()
        optimizer.zero_grad()
        pred = model(X_inv_tr_d, X_basis_tr_d)
        loss = criterion(pred, Y_a_tr_d)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 400:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_inv_val_d, X_basis_val_d)
                val_loss = criterion(val_pred, Y_a_val_d)
            print(f'Epoch [{epoch:3d}/400] - Train Loss: {loss.item():.6e} | Val Loss: {val_loss.item():.6e}')
    model.eval()
    held_out_file = Path('data/benchmark_dns/square_duct_dns_Reb7000_Retau220.csv')
    if not held_out_file.exists():
        held_out_file = Path('data/benchmark_dns/square_duct_dns_Re2200.csv')
    if held_out_file.exists():
        held_data = []
        with open(held_out_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                held_data.append(row)
        N_h = len(held_data)
        grad_U_h = np.zeros((N_h, 3, 3))
        omega_h = np.zeros(N_h)
        a_true_h = np.zeros((N_h, 3, 3))
        for idx, row in enumerate(held_data):
            for i in range(3):
                for j in range(3):
                    grad_U_h[idx, i, j] = float(row[f'grad_U_{i}{j}'])
            omega_h[idx] = float(row['omega'])
            a_true_h[idx, 0, 0] = float(row['a_11'])
            a_true_h[idx, 1, 1] = float(row['a_22'])
            a_true_h[idx, 2, 2] = float(row['a_33'])
            a_true_h[idx, 0, 1] = a_true_h[idx, 1, 0] = float(row['a_12'])
            a_true_h[idx, 0, 2] = a_true_h[idx, 2, 0] = float(row['a_13'])
            a_true_h[idx, 1, 2] = a_true_h[idx, 2, 1] = float(row['a_23'])
        inv_h, T_h = comp_inv_and_bases(grad_U_h, omega_h)
        X_inv_h = torch.tensor(inv_h, dtype=torch.float32).to(device)
        X_basis_h = torch.tensor(T_h, dtype=torch.float32).to(device)
        Y_a_h = torch.tensor(a_true_h, dtype=torch.float32).to(device)
        with torch.no_grad():
            held_pred = model(X_inv_h, X_basis_h)
            held_loss = criterion(held_pred, Y_a_h)
            held_frob = torch.mean(torch.norm(held_pred - Y_a_h, dim=(1, 2))).item()
        print(f'Held-Out Generalization (Re_b = 7000 | Re_h = 3500 | Re_tau = 220): Loss = {held_loss.item():.6e} | Frob = {held_frob:.6e}')
    model_cpu = model.to('cpu')
    torch.save(model_cpu.state_dict(), 'tbnn_weights.pth')
    dummy_inv = torch.zeros((1, 5), dtype=torch.float32)
    dummy_bases = torch.zeros((1, 10, 3, 3), dtype=torch.float32)
    dummy_bases[:, 0, 0, 0] = 1.0
    dummy_bases[:, 0, 1, 1] = -1.0
    traced = torch.jit.trace(model_cpu, (dummy_inv, dummy_bases))
    traced.save('tbnn_model.pt')
    case_pt = Path('verification_squareDuct/constant/tbnn_model.pt')
    case_pt.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(case_pt))
    print(f'[+] Saved trained model to {case_pt}')

if __name__ == '__main__':
    train()
