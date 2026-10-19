import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_foam_vec(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunk = content[content.find('internalField'):]
    data_str = chunk[chunk.find('(') + 1:chunk.rfind(')')].strip()
    vectors = []
    for line in data_str.split('\n'):
        parts = line.strip().strip('()').split()
        if len(parts) == 3:
            try:
                vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                pass
    return np.array(vectors)

def parse_foam_scalar(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunk = content[content.find('internalField'):]
    data_str = chunk[chunk.find('(') + 1:chunk.find(')', chunk.find('('))].strip()
    return np.array([float(x) for x in data_str.split()])

def comp_inv_and_bases_np(grad_U, omega):
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
    return inv, T

def analyze_timescale_and_invariants(repo_root: Path):
    dns_file = repo_root / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb4410_Retau150.csv'
    if not dns_file.exists():
        dns_file = repo_root / 'data' / 'benchmark_dns' / 'square_duct_dns_Re1500.csv'
    df_dns = pd.read_csv(dns_file)
    N_dns = len(df_dns)
    grad_U_dns = np.zeros((N_dns, 3, 3))
    for i in range(3):
        for j in range(3):
            grad_U_dns[:, i, j] = df_dns[f'grad_U_{i}{j}'].values
    omega_dns = df_dns['omega'].values
    tau_dns = np.minimum(5.0, 1.0 / np.maximum(omega_dns, 1e-4))
    inv_dns, T_dns = comp_inv_and_bases_np(grad_U_dns, omega_dns)

    case_dir = repo_root / 'verification_squareDuct'
    u_file = case_dir / '300' / 'U'
    if not u_file.exists():
        u_file = case_dir / '200' / 'U'
    omega_file = case_dir / '300' / 'omega'
    if not omega_file.exists():
        omega_file = case_dir / '200' / 'omega'

    u_cfd = parse_foam_vec(u_file)
    omega_cfd = parse_foam_scalar(omega_file)
    tau_cfd = np.minimum(5.0, 1.0 / np.maximum(omega_cfd, 1e-4))

    Nx_c, Ny_c, Nz_c = 50, 20, 20
    u_3d = u_cfd.reshape(Nz_c, Ny_c, Nx_c, 3)
    dx = 1.0 / Nx_c
    dy = 0.1 / Ny_c
    dz = 0.1 / Nz_c

    grad_U_cfd = np.zeros((Nz_c, Ny_c, Nx_c, 3, 3))
    for c in range(3):
        grad_U_cfd[:, :, :, c, 0] = np.gradient(u_3d[:, :, :, c], axis=2) / dx
        grad_U_cfd[:, :, :, c, 1] = np.gradient(u_3d[:, :, :, c], axis=1) / dy
        grad_U_cfd[:, :, :, c, 2] = np.gradient(u_3d[:, :, :, c], axis=0) / dz

    grad_U_flat = grad_U_cfd.reshape(-1, 3, 3)
    inv_cfd, T_cfd = comp_inv_and_bases_np(grad_U_flat, omega_cfd)

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
    ax2.scatter(inv_cfd[:, 0], inv_cfd[:, 1], color='#d62728', alpha=0.45, s=8, label='CFD Operating Points (All Cells)')
    ax2.set_xlabel(r'$I_1 = \mathrm{tr}(\mathbf{s}^2)$', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'$I_2 = \mathrm{tr}(\mathbf{w}^2)$', fontsize=11, fontweight='bold')
    ax2.set_title(r'Invariant Phase Space $(I_1, I_2)$ & Operating Envelope', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(frameon=True, fontsize=10)

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
    print('| Invariant | DNS Training Range | CFD Operating Range | In-Distribution Coverage |')
    print('| :--- | :---: | :---: | :---: |')
    for i in range(5):
        c = coverage[f'I_{i+1}']
        print(f"| I_{i+1} | [{c['dns_min']:+.3e}, {c['dns_max']:+.3e}] | [{c['cfd_min']:+.3e}, {c['cfd_max']:+.3e}] | **{c['coverage_percent']:.1f}%** |")
    print('================================================================================')
    print(f'[+] Generated diagnostic distribution figure: {fig_path}')

    return coverage, basis_comparison

if __name__ == '__main__':
    repo = Path(__file__).resolve().parent.parent.parent
    analyze_timescale_and_invariants(repo)
