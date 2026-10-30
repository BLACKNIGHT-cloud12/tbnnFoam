import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tbnn_tensor_math import comp_inv_and_bases

def train_np():
    csv_path = None
    if 'TBNN_DNS_DIR' in os.environ:
        cand = Path(os.environ['TBNN_DNS_DIR']) / 'square_duct_dns_Reb4410_Retau150.csv'
        csv_path = cand if cand.exists() else None
    if not csv_path or not csv_path.exists():
        csv_path = _REPO_ROOT / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb4410_Retau150.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Cannot find dataset at {csv_path}.\n'
            'To run this fallback training script, obtain the benchmark dataset per data/benchmark_dns/README.md '
            'or set TBNN_DNS_DIR.'
        )
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
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp.fit(X, g_targets)
    print(f'Fitted MLP score: {mlp.score(X, g_targets):.4f}')
    export_dir = _REPO_ROOT / 'python' / 'legacy_exports'
    export_dir.mkdir(parents=True, exist_ok=True)
    weights_dict = {'inv_mean': inv_mean.tolist(), 'inv_std': inv_std.tolist()}
    layer_map = [0, 2, 4, 6]
    for idx, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        pytorch_idx = layer_map[idx]
        weights_dict[f'{pytorch_idx}.weight'] = W.T.tolist()
        weights_dict[f'{pytorch_idx}.bias'] = b.tolist()
    out_file = export_dir / 'weights_numpy_fallback.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)
    print(f'Exported legacy fallback weights to {out_file}')
if __name__ == '__main__':
    train_np()
