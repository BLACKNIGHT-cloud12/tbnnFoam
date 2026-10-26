import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from pathlib import Path

# Single source of truth for the tensor algebra lives in tbnn_tensor_math;
# these re-exports keep the historical import path (tests/test_parity.py)
# working.
from tbnn_tensor_math import comp_inv_and_bases, cardano_eigvals, project_realizable

__all__ = ['TensorBasisNeuralNetwork', 'comp_inv_and_bases', 'cardano_eigvals',
           'project_realizable', 'train']

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

def train():
    # Fixed seeds: the shipped weights are reproducible from this script on
    # the pinned dependency versions in requirements.txt (CPU determinism;
    # exact bit-equality across CUDA/platforms is not guaranteed).
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # The shipped model is trained on the canonical Re_b=4410 benchmark only;
    # no fallback dataset resolution. Legacy dataset names (Re1500/Re2200)
    # were removed with the re-audited benchmark.
    csv_file = Path(__file__).resolve().parent / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb4410_Retau150.csv'
    if not csv_file.exists():
        csv_file = Path('data/benchmark_dns/square_duct_dns_Reb4410_Retau150.csv')
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
    # Reflection-degenerate split: all five invariants are even in the
    # rotation tensor, so quadrant 1 shares identical inputs with the
    # training quadrants; it tests the odd-in-w basis sign structure
    # (T2, T5, T7, T8, T10), not independent-flow generalization.
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
    held_out_file = Path(__file__).resolve().parent / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb7000_Retau220.csv'
    if not held_out_file.exists():
        held_out_file = Path('data/benchmark_dns/square_duct_dns_Reb7000_Retau220.csv')
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
    traced.save(str(case_pt))
    print(f'[+] Saved trained model to {case_pt}')
    from tbnn_export import export_cpp_weights
    export_cpp_weights(model_cpu)

if __name__ == '__main__':
    train()
