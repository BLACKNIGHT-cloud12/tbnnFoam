import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

def process_dat_file(dat_path, out_csv_path, h=0.5, Ub=1.0, nu=0.0006666667, target_grid=65):
    with open(dat_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_lines = []
    zone_line = ''
    for line in lines:
        s = line.strip()
        if s.startswith('ZONE') or 'I=' in s:
            zone_line = s
        if s and not s.startswith('TITLE') and not s.startswith('VARIABLES') and not s.startswith('ZONE') and not s.startswith('STRANDID') and not s.startswith('DATAPACKING') and not s.startswith('DT=') and not s.startswith('"') and not s.startswith('I='):
            data_lines.append(s)
    I_val = 64
    J_val = 64
    for token in zone_line.replace(',', ' ').split():
        if token.startswith('I='):
            I_val = int(token.split('=')[1])
        elif token.startswith('J='):
            J_val = int(token.split('=')[1])
    arr = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 12:
            arr.append([float(p) for p in parts[:12]])
    raw = np.array(arr)
    zn = raw[:, 0].reshape(I_val, J_val)
    yn = raw[:, 1].reshape(I_val, J_val)
    u_ub = raw[:, 2].reshape(I_val, J_val)
    u_utau = raw[:, 3].reshape(I_val, J_val)
    v_ub = raw[:, 4].reshape(I_val, J_val)
    w_ub = raw[:, 5].reshape(I_val, J_val)
    u2_utau2 = raw[:, 6].reshape(I_val, J_val)
    v2_utau2 = raw[:, 7].reshape(I_val, J_val)
    w2_utau2 = raw[:, 8].reshape(I_val, J_val)
    uv_utau2 = raw[:, 9].reshape(I_val, J_val)
    uw_utau2 = raw[:, 10].reshape(I_val, J_val)
    vw_utau2 = raw[:, 11].reshape(I_val, J_val)
    valid = (u_utau > 1e-4)
    utau_ub = np.median(u_ub[valid] / u_utau[valid])
    utau = utau_ub * Ub
    uu = u2_utau2 * (utau**2)
    vv = v2_utau2 * (utau**2)
    ww = w2_utau2 * (utau**2)
    uv = uv_utau2 * (utau**2)
    uw = uw_utau2 * (utau**2)
    vw = vw_utau2 * (utau**2)
    U_q = u_ub * Ub
    V_q = v_ub * Ub
    W_q = w_ub * Ub
    z_1d_half = zn[:, 0]
    y_1d_half = yn[0, :]
    z_1d_full = np.concatenate([z_1d_half, -z_1d_half[::-1]])
    y_1d_full = np.concatenate([y_1d_half, -y_1d_half[::-1]])
    U_full = np.block([[U_q, U_q[:, ::-1]], [U_q[::-1, :], U_q[::-1, ::-1]]])
    V_full = np.block([[V_q, -V_q[:, ::-1]], [V_q[::-1, :], -V_q[::-1, ::-1]]])
    W_full = np.block([[W_q, W_q[:, ::-1]], [-W_q[::-1, :], -W_q[::-1, ::-1]]])
    uu_full = np.block([[uu, uu[:, ::-1]], [uu[::-1, :], uu[::-1, ::-1]]])
    vv_full = np.block([[vv, vv[:, ::-1]], [vv[::-1, :], vv[::-1, ::-1]]])
    ww_full = np.block([[ww, ww[:, ::-1]], [ww[::-1, :], ww[::-1, ::-1]]])
    uv_full = np.block([[uv, -uv[:, ::-1]], [uv[::-1, :], -uv[::-1, ::-1]]])
    uw_full = np.block([[uw, uw[:, ::-1]], [-uw[::-1, :], -uw[::-1, ::-1]]])
    vw_full = np.block([[vw, -vw[:, ::-1]], [-vw[::-1, :], vw[::-1, ::-1]]])
    y_target = np.linspace(-1.0, 1.0, target_grid)
    z_target = np.linspace(-1.0, 1.0, target_grid)
    Z_tar, Y_tar = np.meshgrid(z_target, y_target, indexing='ij')
    pts_eval = np.column_stack([Z_tar.flatten(), Y_tar.flatten()])
    interp_u = RegularGridInterpolator((z_1d_full, y_1d_full), U_full, bounds_error=False, fill_value=0.0)
    interp_v = RegularGridInterpolator((z_1d_full, y_1d_full), V_full, bounds_error=False, fill_value=0.0)
    interp_w = RegularGridInterpolator((z_1d_full, y_1d_full), W_full, bounds_error=False, fill_value=0.0)
    interp_uu = RegularGridInterpolator((z_1d_full, y_1d_full), uu_full, bounds_error=False, fill_value=0.0)
    interp_vv = RegularGridInterpolator((z_1d_full, y_1d_full), vv_full, bounds_error=False, fill_value=0.0)
    interp_ww = RegularGridInterpolator((z_1d_full, y_1d_full), ww_full, bounds_error=False, fill_value=0.0)
    interp_uv = RegularGridInterpolator((z_1d_full, y_1d_full), uv_full, bounds_error=False, fill_value=0.0)
    interp_uw = RegularGridInterpolator((z_1d_full, y_1d_full), uw_full, bounds_error=False, fill_value=0.0)
    interp_vw = RegularGridInterpolator((z_1d_full, y_1d_full), vw_full, bounds_error=False, fill_value=0.0)
    U_g = interp_u(pts_eval).reshape(target_grid, target_grid)
    V_g = interp_v(pts_eval).reshape(target_grid, target_grid)
    W_g = interp_w(pts_eval).reshape(target_grid, target_grid)
    uu_g = np.clip(interp_uu(pts_eval).reshape(target_grid, target_grid), 0.0, None)
    vv_g = np.clip(interp_vv(pts_eval).reshape(target_grid, target_grid), 0.0, None)
    ww_g = np.clip(interp_ww(pts_eval).reshape(target_grid, target_grid), 0.0, None)
    uv_g = interp_uv(pts_eval).reshape(target_grid, target_grid)
    uw_g = interp_uw(pts_eval).reshape(target_grid, target_grid)
    vw_g = interp_vw(pts_eval).reshape(target_grid, target_grid)
    k_g = 0.5 * (uu_g + vv_g + ww_g)
    k_g = np.clip(k_g, 1e-7, None)
    a_11 = uu_g / (2.0 * k_g) - (1.0 / 3.0)
    a_22 = vv_g / (2.0 * k_g) - (1.0 / 3.0)
    a_33 = ww_g / (2.0 * k_g) - (1.0 / 3.0)
    a_12 = uv_g / (2.0 * k_g)
    a_13 = uw_g / (2.0 * k_g)
    a_23 = vw_g / (2.0 * k_g)
    Y_phys = Y_tar * h
    Z_phys = Z_tar * h
    wall_dist = np.minimum(h - np.abs(Y_phys), h - np.abs(Z_phys))
    wall_dist = np.maximum(wall_dist, 1e-5)
    omega_g = np.sqrt(k_g) / (0.09**0.25 * 0.07 * (2.0 * h)) + 6.0 * nu / (0.075 * wall_dist**2)
    dy = np.gradient(Y_phys, axis=1)
    dz = np.gradient(Z_phys, axis=0)
    grad_U_01 = np.gradient(U_g, axis=1) / dy
    grad_U_02 = np.gradient(U_g, axis=0) / dz
    grad_U_11 = np.gradient(V_g, axis=1) / dy
    grad_U_12 = np.gradient(V_g, axis=0) / dz
    grad_U_21 = np.gradient(W_g, axis=1) / dy
    grad_U_22 = np.gradient(W_g, axis=0) / dz
    N_tot = target_grid * target_grid
    df = pd.DataFrame({
        'y': Y_phys.flatten(),
        'z': Z_phys.flatten(),
        'U': U_g.flatten(),
        'V': V_g.flatten(),
        'W': W_g.flatten(),
        'k': k_g.flatten(),
        'omega': omega_g.flatten(),
        'grad_U_00': np.zeros(N_tot),
        'grad_U_01': grad_U_01.flatten(),
        'grad_U_02': grad_U_02.flatten(),
        'grad_U_10': np.zeros(N_tot),
        'grad_U_11': grad_U_11.flatten(),
        'grad_U_12': grad_U_12.flatten(),
        'grad_U_20': np.zeros(N_tot),
        'grad_U_21': grad_U_21.flatten(),
        'grad_U_22': grad_U_22.flatten(),
        'a_11': a_11.flatten(),
        'a_22': a_22.flatten(),
        'a_33': a_33.flatten(),
        'a_12': a_12.flatten(),
        'a_13': a_13.flatten(),
        'a_23': a_23.flatten()
    })
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    print(f'Full-duct DNS: {dat_path} -> {out_csv_path} ({len(df)} points, y in [{df["y"].min():.2f}, {df["y"].max():.2f}])')

if __name__ == '__main__':
    raw_dir = Path('data/benchmark_dns/raw_dns')
    out_dir = Path('data/benchmark_dns')
    process_dat_file(raw_dir / 'plotyz_Retau150.dat', out_dir / 'square_duct_dns_Re1500.csv', h=0.5, Ub=1.0, nu=0.000226757, target_grid=65)
    process_dat_file(raw_dir / 'plotyz_Retau220.dat', out_dir / 'square_duct_dns_Re2200.csv', h=0.5, Ub=1.0, nu=0.000142857, target_grid=65)
