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
        self.mlp = nn.Sequential(nn.Linear(num_invariants, hidden_dim), nn.LeakyReLU(0.1), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1), nn.Linear(hidden_dim, num_bases))

    def forward(self, invariants: torch.Tensor, tensor_bases: torch.Tensor) -> torch.Tensor:
        invariants = torch.clamp(invariants, -50.0, 50.0)
        g = self.mlp(invariants)
        g_expanded = g.unsqueeze(-1).unsqueeze(-1)
        anisotropy = torch.sum(g_expanded * tensor_bases, dim=1)
        return anisotropy

def comp_inv_and_bases(grad_U, omega):
    N = grad_U.shape[0]
    tau = 1.0 / np.clip(omega, 0.0001, 1000000.0)
    S = 0.5 * (grad_U + np.swapaxes(grad_U, 1, 2))
    W = 0.5 * (grad_U - np.swapaxes(grad_U, 1, 2))
    s = S * tau[:, None, None]
    w = W * tau[:, None, None]
    s2 = np.matmul(s, s)
    w2 = np.matmul(w, w)
    I1 = np.trace(s2, axis1=1, axis2=2)[:, None]
    I2 = np.trace(w2, axis1=1, axis2=2)[:, None]
    I3 = np.trace(np.matmul(s2, s), axis1=1, axis2=2)[:, None]
    I4 = np.trace(np.matmul(w2, s), axis1=1, axis2=2)[:, None]
    I5 = np.trace(np.matmul(w2, s2), axis1=1, axis2=2)[:, None]
    inv = np.hstack([I1, I2, I3, I4, I5])
    I = np.eye(3)[None, :, :]
    T = np.zeros((N, 10, 3, 3))
    T[:, 0] = s
    T[:, 1] = np.matmul(s, w) - np.matmul(w, s)
    T[:, 2] = s2 - 1.0 / 3.0 * I1[:, :, None] * I
    T[:, 3] = w2 - 1.0 / 3.0 * I2[:, :, None] * I
    return (inv, T)

def train():
    csv_file = Path('data/benchmark_dns/square_duct_dns_Re1500.csv')
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    N = len(data)
    print(f'Loaded {N} points from {csv_file}')
    grad_U = np.zeros((N, 3, 3))
    omega = np.zeros(N)
    a_true = np.zeros((N, 3, 3))
    for idx, row in enumerate(data):
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
    X_inv = torch.tensor(inv, dtype=torch.float32)
    X_basis = torch.tensor(T, dtype=torch.float32)
    Y_a = torch.tensor(a_true, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    model = TensorBasisNeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    X_inv_d = X_inv.to(device)
    X_basis_d = X_basis.to(device)
    Y_a_d = Y_a.to(device)
    for epoch in range(1, 401):
        model.train()
        optimizer.zero_grad()
        pred = model(X_inv_d, X_basis_d)
        loss = criterion(pred, Y_a_d)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == 400:
            print(f'Epoch [{epoch:3d}/400] - Loss: {loss.item():.6e}')
    model.eval()
    model_cpu = model.to('cpu')
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
