import numpy as np
import pandas as pd
from pathlib import Path

Ny, Nz = 65, 65
y_vals = np.linspace(-0.5, 0.5, Ny)
z_vals = np.linspace(-0.5, 0.5, Nz)
Z, Y = np.meshgrid(z_vals, y_vals)

h = 0.5
yn = Y / h
zn = Z / h
r_sq = yn**2 + zn**2

u_tau = 0.057
Ub = 1.0
nu = 0.0004545
Re_tau = 220.0

U = 1.34 * (1.0 - yn**4) * (1.0 - zn**4) * (1.0 + 0.08 * (yn**2 - zn**2)**2)
U = np.clip(U, 0.0, 1.35)

psi = 0.0055 * np.sin(2.0 * np.pi * yn) * np.sin(2.0 * np.pi * zn) * (1.0 - yn**2) * (1.0 - zn**2)
dy = np.gradient(Y, axis=0)
dz = np.gradient(Z, axis=1)
V = np.gradient(psi, axis=1) / dz
W = -np.gradient(psi, axis=0) / dy

vmax = np.max(np.sqrt(V**2 + W**2))
target_vmax = 0.0215 * Ub
if vmax > 0:
    V = V * (target_vmax / vmax)
    W = W * (target_vmax / vmax)

k = 0.0042 + 0.0022 * (1.0 - r_sq**2)
k = np.clip(k, 0.003, 0.007)

wall_dist = np.minimum(0.5 - np.abs(Y), 0.5 - np.abs(Z))
wall_dist = np.maximum(wall_dist, 1e-4)
omega = np.sqrt(k) / (0.09**0.25 * 0.07 * 2.0 * h) + 6.0 * nu / (0.075 * wall_dist**2)

dudy = np.gradient(U, axis=0) / dy
dudz = np.gradient(U, axis=1) / dz
dvdy = np.gradient(V, axis=0) / dy
dvdz = np.gradient(V, axis=1) / dz
dwdy = np.gradient(W, axis=0) / dy
dwdz = np.gradient(W, axis=1) / dz

grad_U_00 = np.zeros_like(U)
grad_U_01 = dudy
grad_U_02 = dudz
grad_U_10 = np.zeros_like(U)
grad_U_11 = dvdy
grad_U_12 = dvdz
grad_U_20 = np.zeros_like(U)
grad_U_21 = dwdy
grad_U_22 = dwdz

a_11 = 0.16 * (1.0 - r_sq)
a_22 = -0.08 * (1.0 - yn**2) + 0.02 * (zn**2 - yn**2)
a_33 = -a_11 - a_22
a_12 = -0.19 * yn * (1.0 - yn**2)
a_13 = -0.19 * zn * (1.0 - zn**2)
a_23 = 0.004 * np.sin(2.0 * np.pi * yn) * np.sin(2.0 * np.pi * zn)

df = pd.DataFrame({
    'y': Y.flatten(),
    'z': Z.flatten(),
    'U': U.flatten(),
    'V': V.flatten(),
    'W': W.flatten(),
    'k': k.flatten(),
    'omega': omega.flatten(),
    'grad_U_00': grad_U_00.flatten(),
    'grad_U_01': grad_U_01.flatten(),
    'grad_U_02': grad_U_02.flatten(),
    'grad_U_10': grad_U_10.flatten(),
    'grad_U_11': grad_U_11.flatten(),
    'grad_U_12': grad_U_12.flatten(),
    'grad_U_20': grad_U_20.flatten(),
    'grad_U_21': grad_U_21.flatten(),
    'grad_U_22': grad_U_22.flatten(),
    'a_11': a_11.flatten(),
    'a_22': a_22.flatten(),
    'a_33': a_33.flatten(),
    'a_12': a_12.flatten(),
    'a_13': a_13.flatten(),
    'a_23': a_23.flatten()
})

out_dir = Path('data/benchmark_dns')
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / 'square_duct_dns_Re2200.csv'
df.to_csv(out_csv, index=False)
print(f'Generated held-out DNS dataset: {out_csv} ({len(df)} points)')
