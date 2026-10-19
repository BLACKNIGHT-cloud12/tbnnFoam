import json
from pathlib import Path
import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

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

def train_np():
    csv_path = Path('data/benchmark_dns/square_duct_dns_Reb4410_Retau150.csv')
    if not csv_path.exists():
        csv_path = Path('data/benchmark_dns/square_duct_dns_Re1500.csv')
    df = pd.read_csv(csv_path)
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
    invariants, T = comp_inv_and_bases(grad_U, omega)
    inv_mean = np.mean(invariants, axis=0)
    inv_std = np.std(invariants, axis=0) + 1e-06
    X = (invariants - inv_mean) / inv_std
    T_flat = T.reshape(N, 10, 9)
    a_flat = a_true.reshape(N, 9)
    g_targets = np.zeros((N, 10))
    for i in range(N):
        M = T_flat[i]
        g_targets[i] = np.linalg.pinv(M.T) @ a_flat[i]
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(30, 30, 30), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp.fit(X, g_targets)
    print(f'Fitted MLP score: {mlp.score(X, g_targets):.4f}')
    export_dir = Path('export_weights')
    export_dir.mkdir(parents=True, exist_ok=True)
    weights_dict = {'inv_mean': inv_mean.tolist(), 'inv_std': inv_std.tolist()}
    layer_map = [0, 2, 4, 6]
    for idx, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        pytorch_idx = layer_map[idx]
        weights_dict[f'{pytorch_idx}.weight'] = W.T.tolist()
        weights_dict[f'{pytorch_idx}.bias'] = b.tolist()
    out_file = export_dir / 'weights.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)
    print(f'Exported weights to {out_file}')
if __name__ == '__main__':
    train_np()
