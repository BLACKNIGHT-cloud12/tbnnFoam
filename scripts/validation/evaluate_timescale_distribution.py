import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.validation.foam_field_io import parse_foam_vec, parse_foam_scalar
from tbnn_tensor_math import comp_inv_and_bases

def analyze_timescale_and_invariants(repo_root: Path):
    dns_file = repo_root / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb4410_Retau150.csv'
    if not dns_file.exists():
        raise FileNotFoundError(f'Canonical DNS dataset missing: {dns_file}')
    df_dns = pd.read_csv(dns_file)
    N_dns = len(df_dns)
    grad_U_dns = np.zeros((N_dns, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_U_dns[:, i, j] = df_dns[f'grad_U_{i}{j}'].values
    omega_dns = df_dns['omega'].values
    tau_dns = np.minimum(5.0, 1.0 / np.maximum(omega_dns, 1e-4))
    inv_dns, T_dns = comp_inv_and_bases(grad_U_dns, omega_dns)

    case_dir = repo_root / 'verification_squareDuct'
    # Committed result directory is the canonical evaluation input; the
    # legacy '200' fallback pointed at a directory that no longer exists.
    u_file = case_dir / '300' / 'U'
    omega_file = case_dir / '300' / 'omega'
    if not u_file.exists():
        raise FileNotFoundError(f'Committed solution missing: {u_file}')
    if not omega_file.exists():
        raise FileNotFoundError(f'Committed omega field missing: {omega_file}')

    u_cfd = parse_foam_vec(u_file)
    omega_cfd = parse_foam_scalar(omega_file)
    tau_cfd = np.minimum(5.0, 1.0 / np.maximum(omega_cfd, 1e-4))

    c_file = case_dir / '0' / 'C'
    if c_file.exists():
        c_raw = parse_foam_vec(c_file)
        x_u = np.unique(np.round(c_raw[:, 0], 6))
        y_u = np.unique(np.round(c_raw[:, 1], 6))
        z_u = np.unique(np.round(c_raw[:, 2], 6))
        Nx_c, Ny_c, Nz_c = len(x_u), len(y_u), len(z_u)
        dx = float(x_u[1] - x_u[0])
        dy = float(y_u[1] - y_u[0])
        dz = float(z_u[1] - z_u[0])
    else:
        Nx_c, Ny_c, Nz_c = 50, 20, 20
        dx = 1.0 / Nx_c
        dy = 0.1 / Ny_c
        dz = 0.1 / Nz_c

    u_3d = u_cfd.reshape(Nz_c, Ny_c, Nx_c, 3)
    grad_U_cfd = np.zeros((Nz_c, Ny_c, Nx_c, 3, 3))
    for c in range(3):
        grad_U_cfd[:, :, :, c, 0] = np.gradient(u_3d[:, :, :, c], axis=2) / dx
        grad_U_cfd[:, :, :, c, 1] = np.gradient(u_3d[:, :, :, c], axis=1) / dy
        grad_U_cfd[:, :, :, c, 2] = np.gradient(u_3d[:, :, :, c], axis=0) / dz

    grad_U_flat = grad_U_cfd.reshape(-1, 3, 3)
    inv_cfd, T_cfd = comp_inv_and_bases(grad_U_flat, omega_cfd)

    coverage = {}
    for i in range(5):
        d_min = float(np.min(inv_dns[:, i]))
        d_max = float(np.max(inv_dns[:, i]))
        c_min = float(np.min(inv_cfd[:, i]))
        c_max = float(np.max(inv_cfd[:, i]))
        pct = float(np.mean((inv_cfd[:, i] >= d_min) & (inv_cfd[:, i] <= d_max)) * 100.0)
        coverage[f'I_{i+1}'] = {
            'dns_min': d_min,
            'dns_max': d_max,
            'cfd_min': c_min,
            'cfd_max': c_max,
            'coverage_percent': pct
        }

    mu_dns = np.mean(inv_dns, axis=0)
    std_dns = np.std(inv_dns, axis=0) + 1e-12
    z_dns = (inv_dns - mu_dns) / std_dns
    z_cfd = (inv_cfd - mu_dns) / std_dns
    tree = cKDTree(z_dns)
    d_cfd, _ = tree.query(z_cfd, k=1)
    joint_diagnostic = {
        'knn_dist_median': float(np.median(d_cfd)),
        'knn_dist_mean': float(np.mean(d_cfd)),
        'knn_dist_95pct': float(np.percentile(d_cfd, 95)),
        'knn_dist_max': float(np.max(d_cfd))
    }
    coverage['joint_diagnostic'] = joint_diagnostic

    basis_comparison = {}
    for n in range(10):
        norm_dns = np.linalg.norm(T_dns[:, n], axis=(1, 2))
        norm_cfd = np.linalg.norm(T_cfd[:, n], axis=(1, 2))
        basis_comparison[f'T_{n}'] = {
            'dns_mean': float(np.mean(norm_dns)),
            'dns_max': float(np.max(norm_dns)),
            'cfd_mean': float(np.mean(norm_cfd)),
            'cfd_max': float(np.max(norm_cfd))
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor('#ffffff')

    ax1 = axes[0, 0]
    log_omega_dns = np.log10(np.clip(omega_dns, 1e-1, 1e8))
    log_omega_cfd = np.log10(np.clip(omega_cfd, 1e-1, 1e8))
    ax1.hist(log_omega_dns, bins=40, density=True, alpha=0.55, color='#1f77b4', label=f'DNS (Wall-dist Est., N={len(omega_dns)})')
    ax1.hist(log_omega_cfd, bins=40, density=True, alpha=0.55, color='#ff7f0e', label=f'CFD (OpenFOAM SST, N={len(omega_cfd)})')
    ax1.set_xlabel(r'$\log_{10}(\omega)$ [$\mathrm{s}^{-1}$]', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax1.set_title(r'Turbulent Specific Dissipation Rate ($\omega$) Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(frameon=True, fontsize=10)

    ax2 = axes[0, 1]
    ax2.scatter(inv_dns[:, 0], inv_dns[:, 1], color='#1f77b4', alpha=0.25, s=12, label='DNS Training Points')
    sc = ax2.scatter(inv_cfd[:, 0], inv_cfd[:, 1], c=d_cfd, cmap='plasma', alpha=0.55, s=10, label='CFD Operating Cells')
    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label('Standardized 5D kNN Distance', fontsize=9, fontweight='bold')
    ax2.set_xlabel(r'$I_1 = \mathrm{tr}(\mathbf{s}^2)$', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'$I_2 = \mathrm{tr}(\mathbf{w}^2)$', fontsize=11, fontweight='bold')
    ax2.set_title(r'Invariant Phase Space $(I_1, I_2)$ & Joint Distribution Shift', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True, fontsize=9, loc='lower left')

    ax3 = axes[1, 0]
    inv_labels = [r'$I_1$', r'$I_2$', r'$I_3$', r'$I_4$', r'$I_5$']
    dns_ranges = [coverage[f'I_{i+1}']['dns_max'] - coverage[f'I_{i+1}']['dns_min'] for i in range(5)]
    cfd_spans = [coverage[f'I_{i+1}']['cfd_max'] - coverage[f'I_{i+1}']['cfd_min'] for i in range(5)]
    ratio = [cfd_spans[i] / (dns_ranges[i] + 1e-12) * 100.0 for i in range(5)]
    bars = ax3.bar(inv_labels, ratio, color='#2ca02c', alpha=0.8, edgecolor='black', width=0.5)
    ax3.axhline(100.0, color='red', linestyle='--', linewidth=1.5, label='100% DNS Range Limit')
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, 120)
    ax3.set_xlabel('Cayley-Hamilton Invariants', fontsize=11, fontweight='bold')
    ax3.set_ylabel('CFD Domain Span / DNS Domain Span [%]', fontsize=11, fontweight='bold')
    ax3.set_title('CFD Operating Envelope Span Relative to DNS Training Support', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax3.legend(frameon=True, fontsize=10)

    ax4 = axes[1, 1]
    basis_indices = np.arange(10)
    width = 0.35
    dns_basis_max = [basis_comparison[f'T_{n}']['dns_max'] for n in range(10)]
    cfd_basis_max = [basis_comparison[f'T_{n}']['cfd_max'] for n in range(10)]
    ax4.bar(basis_indices - width/2, dns_basis_max, width, label='DNS Max Norm', color='#1f77b4', alpha=0.75)
    ax4.bar(basis_indices + width/2, cfd_basis_max, width, label='CFD Max Norm', color='#ff7f0e', alpha=0.85)
    ax4.set_yscale('log')
    ax4.set_xticks(basis_indices)
    ax4.set_xticklabels([rf'$\mathbf{{T}}^{{({n+1})}}$' for n in range(10)])
    ax4.set_xlabel('Pope Tensor Basis', fontsize=11, fontweight='bold')
    ax4.set_ylabel(r'Max Frobenius Norm $\|\mathbf{T}^{(n)}\|_F$ [log scale]', fontsize=11, fontweight='bold')
    ax4.set_title('Tensor Basis Frobenius Norm Bound Verification', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4, which='both')
    ax4.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    fig_path = repo_root / 'docs' / 'figures' / 'timescale_and_invariant_distribution.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print('================================================================================')
    print('         TIME-SCALE & INVARIANT DISTRIBUTION SHIFT ANALYSIS (DNS VS SST)        ')
    print('================================================================================')
    print('| Invariant | DNS Training Range | CFD Operating Range | Marginal Range Coverage |')
    print('| :--- | :---: | :---: | :---: |')
    for i in range(5):
        c = coverage[f'I_{i+1}']
        print(f"| I_{i+1} | [{c['dns_min']:+.3e}, {c['dns_max']:+.3e}] | [{c['cfd_min']:+.3e}, {c['cfd_max']:+.3e}] | **{c['coverage_percent']:.1f}%** |")
    print('--------------------------------------------------------------------------------')
    print(f"| **Joint 5D Feature Distance (kNN)** | Median: {joint_diagnostic['knn_dist_median']:.2f} std | Mean: {joint_diagnostic['knn_dist_mean']:.2f} std | 95th Pct: {joint_diagnostic['knn_dist_95pct']:.2f} std | Max: {joint_diagnostic['knn_dist_max']:.2f} std |")
    print('================================================================================')
    print(f'[+] Generated diagnostic distribution figure: {fig_path}')

    return coverage, basis_comparison

if __name__ == '__main__':
    repo = Path(__file__).resolve().parent.parent.parent
    analyze_timescale_and_invariants(repo)
