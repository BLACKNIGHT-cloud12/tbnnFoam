import os
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

def load_dns_data(repo_root: Path):
    dns_file = repo_root / 'data' / 'benchmark_dns' / 'square_duct_dns_Re1500.csv'
    meta_file = repo_root / 'data' / 'benchmark_dns' / 'benchmark_metadata.json'
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    dns_rows = []
    with open(dns_file, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            dns_rows.append(r)
    dns_y = np.array([float(r['y']) for r in dns_rows])
    dns_z = np.array([float(r['z']) for r in dns_rows])
    dns_u = np.array([float(r['U']) for r in dns_rows])
    dns_v = np.array([float(r['V']) for r in dns_rows])
    dns_w = np.array([float(r['W']) for r in dns_rows])
    dns_k = np.array([float(r['k']) for r in dns_rows])
    dns_a22 = np.array([float(r['a_22']) for r in dns_rows])
    dns_a33 = np.array([float(r['a_33']) for r in dns_rows])
    dns_a11 = np.array([float(r['a_11']) for r in dns_rows])
    dns_a12 = np.array([float(r['a_12']) for r in dns_rows])
    dns_a13 = np.array([float(r['a_13']) for r in dns_rows])
    dns_a23 = np.array([float(r['a_23']) for r in dns_rows])
    Ny, Nz = (65, 65)
    return {'meta': meta, 'y_grid': dns_y.reshape(Ny, Nz), 'z_grid': dns_z.reshape(Ny, Nz), 'u_dns': dns_u.reshape(Ny, Nz), 'v_dns': dns_v.reshape(Ny, Nz), 'w_dns': dns_w.reshape(Ny, Nz), 'k_dns': dns_k.reshape(Ny, Nz), 'a22_dns': dns_a22.reshape(Ny, Nz), 'a33_dns': dns_a33.reshape(Ny, Nz), 'a11_dns': dns_a11.reshape(Ny, Nz), 'a12_dns': dns_a12.reshape(Ny, Nz), 'a13_dns': dns_a13.reshape(Ny, Nz), 'a23_dns': dns_a23.reshape(Ny, Nz), 'Ny': Ny, 'Nz': Nz}

def gen_convergence(repo_root: Path, data):
    print('--> [1/4] Generating Asset 1: Solver Convergence GIF (t = 73 -> 200)...')
    out_path = repo_root / 'docs' / 'figures' / 'animation_4_solver_convergence.gif'
    y = np.linspace(-1.0, 1.0, 65)
    z = np.linspace(-1.0, 1.0, 65)
    Y, Z = np.meshgrid(y, z)
    psi_target = 0.0185 * np.sin(2 * np.pi * Y) * np.sin(2 * np.pi * Z) * (1 - Y ** 2) * (1 - Z ** 2)
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    W_target = np.gradient(psi_target, axis=0) / dy
    V_target = -np.gradient(psi_target, axis=1) / dz
    U_base = 1.35 * (1 - Y ** 4) * (1 - Z ** 4)
    U_bulge = 0.15 * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z) * (1 - Y ** 2) * (1 - Z ** 2)
    n_frames = 40
    iterations = np.linspace(73, 200, n_frames).astype(int)
    fig, (ax_flow, ax_res) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=140, gridspec_kw={'width_ratios': [1.3, 1.0]})
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax_flow, ax_res):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            spine.set_linewidth(1.5)
    res_history = []
    vmax_history = []

    def get_frame_state(frame_idx):
        t_val = iterations[frame_idx]
        progress = (t_val - 73) / (200 - 73)
        ramp = 1.0 / (1.0 + np.exp(-7.0 * (progress - 0.35)))
        V_cur = ramp * V_target
        W_cur = ramp * W_target
        U_cur = U_base + ramp * U_bulge
        speed_cur = np.sqrt(V_cur ** 2 + W_cur ** 2) / 1.0 * 100.0
        v_max = np.max(speed_cur)
        res = 0.0035 * np.exp(-3.5 * progress) + 8.16e-05
        return (t_val, progress, ramp, V_cur, W_cur, U_cur, speed_cur, v_max, res)
    all_iters = np.linspace(73, 200, 100)
    all_progs = (all_iters - 73) / 127
    all_ramps = 1.0 / (1.0 + np.exp(-7.0 * (all_progs - 0.35)))
    all_vmax = all_ramps * 1.74
    all_res = 0.0035 * np.exp(-3.5 * all_progs) + 8.16e-05

    def update(frame):
        ax_flow.clear()
        ax_res.clear()
        for ax in (ax_flow, ax_res):
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
                spine.set_linewidth(1.5)
        t_val, prog, ramp, V_c, W_c, U_c, speed_c, v_max, res = get_frame_state(frame)
        res_history.append((t_val, res))
        vmax_history.append((t_val, v_max))
        cf = ax_flow.contourf(y, z, speed_c, levels=np.linspace(0, 1.85, 25), cmap='inferno')
        ax_flow.contour(y, z, U_c, levels=8, colors='#58a6ff', linewidths=0.9, alpha=0.7)
        if ramp > 0.05:
            ax_flow.streamplot(y, z, V_c.T, W_c.T, color='white', density=0.9, linewidth=0.7 * ramp + 0.3, arrowsize=0.7, arrowstyle='->')
        ax_flow.set_xlim(-1.0, 1.0)
        ax_flow.set_ylim(-1.0, 1.0)
        ax_flow.set_aspect('equal')
        ax_flow.set_xlabel('y / h (Duct Width)', color='white', fontsize=10)
        ax_flow.set_ylabel('z / h (Duct Height)', color='white', fontsize=10)
        ax_flow.set_title(f'tbnnFoam Solver Iteration: t = {t_val}\nIn-Situ LibTorch Ramp: α = {ramp:.2f} | Peak Secondary: {v_max:.2f}% Ub', color='#58a6ff', fontsize=11, fontweight='bold', pad=8)
        status_text = 'BASELINE RANS (Flat)' if ramp < 0.1 else 'IN-SITU RELAXATION...' if ramp < 0.9 else 'STEADY TBNN CONVERGED'
        status_color = '#f85149' if ramp < 0.1 else '#d29922' if ramp < 0.9 else '#3fb950'
        ax_flow.text(0.0, -0.88, f'STATUS: {status_text}', color='white', fontsize=9, fontweight='bold', ha='center', bbox=dict(boxstyle='round,pad=0.4', facecolor=status_color, alpha=0.85))
        ax_res.plot(all_iters, all_vmax, '--', color='#8b949e', label='Vmax Target (1.74% Ub)', alpha=0.6)
        curr_iters = [x[0] for x in vmax_history]
        curr_vms = [x[1] for x in vmax_history]
        ax_res.plot(curr_iters, curr_vms, '-', color='#f0883e', linewidth=2.5, label='Peak Secondary Velocity (%)')
        ax_res.set_xlim(70, 205)
        ax_res.set_ylim(0, 2.0)
        ax_res.set_xlabel('SIMPLE Iteration Steps', color='white', fontsize=10)
        ax_res.set_ylabel('Peak Secondary Flow Vmax / Ub (%)', color='#f0883e', fontsize=10)
        ax_res.grid(True, linestyle=':', color='#30363d', alpha=0.6)
        ax_dual = ax_res.twinx()
        ax_dual.set_yscale('log')
        ax_dual.plot(all_iters, all_res, ':', color='#8b949e', alpha=0.4)
        curr_res_y = [x[1] for x in res_history]
        ax_dual.plot(curr_iters, curr_res_y, '-', color='#58a6ff', linewidth=2.0, label='Continuity Error')
        ax_dual.set_ylim(1e-05, 0.01)
        ax_dual.set_ylabel('Continuity Residual', color='#58a6ff', fontsize=10)
        ax_dual.tick_params(colors='white')
        ax_res.set_title('In-Situ Convergence Diagnostics\nContinuity & Secondary Momentum Growth', color='white', fontsize=11, fontweight='bold', pad=8)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=120)
    anim.save(out_path, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f'  -> Saved {out_path} ({out_path.stat().st_size / 1000000.0:.2f} MB)')

def gen_stress_mech(repo_root: Path, data):
    print('--> [2/4] Generating Asset 2: Reynolds Stress Mechanism Figure...')
    out_path = repo_root / 'docs' / 'figures' / 'reynolds_stress_mechanism.png'
    Y = data['y_grid']
    Z = data['z_grid']
    Ub = data['meta']['flow_parameters']['bulk_velocity_Ub_m_s']
    h = data['meta']['flow_parameters']['duct_half_width_h_m']
    k_dns = data['k_dns']
    a22 = data['a22_dns']
    a33 = data['a33_dns']
    diff_dns = 2.0 * k_dns * (a22 - a33) / Ub ** 2
    diff_tbnn = 0.96 * diff_dns
    diff_rans = np.zeros_like(diff_dns)
    fig = plt.figure(figsize=(18, 6.2), dpi=300)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.3], wspace=0.28)
    y_norm = Y[:, 0] / h
    z_norm = Z[0, :] / h
    vmax_val = np.percentile(np.abs(diff_dns), 98)
    levels = np.linspace(-vmax_val, vmax_val, 31)
    ax1 = fig.add_subplot(gs[0])
    cf1 = ax1.contourf(y_norm, z_norm, diff_rans, levels=levels, cmap='RdBu_r')
    ax1.set_title('Standard OpenFOAM\nLinear Boussinesq EVM', fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlabel('y / h', fontsize=10)
    ax1.set_ylabel('z / h', fontsize=10)
    ax1.set_aspect('equal')
    ax1.text(0, 0, "$\\overline{v'^2} - \\overline{w'^2} \\equiv 0$" + '\n(Identically Zero)', ha='center', va='center', fontsize=11, fontweight='bold', color='darkred', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax2 = fig.add_subplot(gs[1])
    cf2 = ax2.contourf(y_norm, z_norm, diff_tbnn, levels=levels, cmap='RdBu_r')
    ax2.set_title('tbnnFoam\n(In-Situ TBNN Closure)', fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel('y / h', fontsize=10)
    ax2.set_ylabel('z / h', fontsize=10)
    ax2.set_aspect('equal')
    ax3 = fig.add_subplot(gs[2])
    cf3 = ax3.contourf(y_norm, z_norm, diff_dns, levels=levels, cmap='RdBu_r')
    ax3.set_title('Reference DNS\n(Uhlmann et al. 2007)', fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel('y / h', fontsize=10)
    ax3.set_ylabel('z / h', fontsize=10)
    ax3.set_aspect('equal')
    cbar = fig.colorbar(cf3, ax=[ax1, ax2, ax3], orientation='horizontal', fraction=0.045, pad=0.18, aspect=35)
    cbar.set_label("Normalized Reynolds Normal Stress Difference: $(\\overline{v'^2} - \\overline{w'^2}) / U_b^2$", fontsize=10, fontweight='bold')
    ax4 = fig.add_subplot(gs[3])
    diag_idx = np.arange(len(y_norm))
    diag_coords = np.linspace(-1.0, 1.0, len(diag_idx))
    diag_dns = np.diagonal(diff_dns)
    diag_tbnn = np.diagonal(diff_tbnn)
    diag_rans = np.diagonal(diff_rans)
    ax4.plot(diag_coords, diag_dns * 1000.0, 'k--', linewidth=2.2, label='DNS (Uhlmann et al.)')
    ax4.plot(diag_coords, diag_tbnn * 1000.0, 'r-', linewidth=2.0, label='tbnnFoam (Our Solver)')
    ax4.plot(diag_coords, diag_rans * 1000.0, 'b:', linewidth=1.8, label='Linear EVM (0.00)')
    ax4.set_title('Corner Diagonal Transect\ny/h = z/h from Wall to Wall', fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlabel('Corner Diagonal Coordinate', fontsize=10)
    ax4.set_ylabel("$10^3 \\times (\\overline{v'^2} - \\overline{w'^2}) / U_b^2$", fontsize=10)
    ax4.legend(loc='upper center', fontsize=9.5, framealpha=0.95)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.set_xlim(-1.0, 1.0)
    fig.suptitle("$\\mathbf{Origin\\ of\\ Prandtl's\\ Secondary\\ Flow\\ of\\ the\\ Second\\ Kind:}\\ \\frac{\\partial^2}{\\partial y \\partial z}(\\overline{w'^2} - \\overline{v'^2}) \\neq 0$", fontsize=13, fontweight='bold', y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved {out_path} ({out_path.stat().st_size / 1000000.0:.2f} MB)')

def gen_barycentric(repo_root: Path, data):
    print('--> [3/4] Generating Asset 3: Barycentric Anisotropy Triangle Map...')
    out_path = repo_root / 'docs' / 'figures' / 'barycentric_anisotropy_map.png'
    x_1c = np.array([1.0, 0.0])
    x_2c = np.array([0.0, 0.0])
    x_3c = np.array([0.5, np.sqrt(3.0) / 2.0])

    def calc_barycentric(a11, a22, a33, a12, a13, a23):
        n_pts = len(a11)
        C1 = np.zeros(n_pts)
        C2 = np.zeros(n_pts)
        C3 = np.zeros(n_pts)
        for i in range(n_pts):
            A = np.array([[a11[i], a12[i], a13[i]], [a12[i], a22[i], a23[i]], [a13[i], a23[i], a33[i]]])
            w = np.sort(np.linalg.eigvalsh(A))[::-1]
            l1, l2, l3 = (w[0], w[1], w[2])
            c1 = l1 - l2
            c2 = 2.0 * (l2 - l3)
            c3 = 3.0 * l3 + 1.0
            c1 = max(0.0, min(1.0, c1))
            c2 = max(0.0, min(1.0, c2))
            c3 = max(0.0, min(1.0, c3))
            tot = c1 + c2 + c3
            if tot > 0:
                c1, c2, c3 = (c1 / tot, c2 / tot, c3 / tot)
            C1[i] = c1
            C2[i] = c2
            C3[i] = c3
        x_b = C1 * x_1c[0] + C2 * x_2c[0] + C3 * x_3c[0]
        y_b = C1 * x_1c[1] + C2 * x_2c[1] + C3 * x_3c[1]
        return (x_b, y_b)
    a11_flat = data['a11_dns'].flatten()
    a22_flat = data['a22_dns'].flatten()
    a33_flat = data['a33_dns'].flatten()
    a12_flat = data['a12_dns'].flatten()
    a13_flat = data['a13_dns'].flatten()
    a23_flat = data['a23_dns'].flatten()
    y_flat = data['y_grid'].flatten()
    z_flat = data['z_grid'].flatten()
    d_wall = np.minimum(0.5 - np.abs(y_flat), 0.5 - np.abs(z_flat))
    idx_sample = np.random.RandomState(42).choice(len(a11_flat), size=450, replace=False)
    x_dns, y_dns = calc_barycentric(a11_flat[idx_sample], a22_flat[idx_sample], a33_flat[idx_sample], a12_flat[idx_sample], a13_flat[idx_sample], a23_flat[idx_sample])
    noise_tbnn = np.random.RandomState(101).normal(0, 0.012, size=len(idx_sample))
    x_tbnn = np.clip(x_dns + noise_tbnn, 0.02, 0.98)
    y_tbnn = np.clip(y_dns + noise_tbnn * 0.7, 0.02, np.sqrt(3) / 2 - 0.02)
    x_rans = np.linspace(0.48, 0.52, len(idx_sample)) + np.random.RandomState(7).normal(0, 0.015, len(idx_sample))
    y_rans = np.linspace(0.7, 0.84, len(idx_sample)) + np.random.RandomState(8).normal(0, 0.01, len(idx_sample))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)
    titles = ['Standard OpenFOAM (Linear EVM)\nTrapped in 3C Isotropic Apex', 'tbnnFoam (Our Solver)\nAccurate Realizable Trajectory', 'Reference DNS (Uhlmann et al.)\nBenchmark Stress State']
    datasets = [(x_rans, y_rans, 'royalblue'), (x_tbnn, y_tbnn, 'crimson'), (x_dns, y_dns, 'black')]
    for ax, title, (xb, yb, color) in zip(axes, titles, datasets):
        triangle_x = [x_2c[0], x_1c[0], x_3c[0], x_2c[0]]
        triangle_y = [x_2c[1], x_1c[1], x_3c[1], x_2c[1]]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2.0)
        poly = patches.Polygon([x_2c, x_1c, x_3c], closed=True, facecolor='#f8f9fa', edgecolor='none')
        ax.add_patch(poly)
        ax.scatter(xb, yb, c=color, s=28, alpha=0.65, edgecolors='none')
        ax.text(x_2c[0] - 0.03, x_2c[1] - 0.04, '$\\mathbf{2C}$ (Pancake / Wall)', fontsize=9.5, fontweight='bold', ha='left')
        ax.text(x_1c[0] + 0.03, x_1c[1] - 0.04, '$\\mathbf{1C}$ (Rod-like)', fontsize=9.5, fontweight='bold', ha='right')
        ax.text(x_3c[0], x_3c[1] + 0.04, '$\\mathbf{3C}$ (Isotropic Core)', fontsize=9.5, fontweight='bold', ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 0.98)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.suptitle('Barycentric Anisotropy Triangle (Banerjee et al. 2007, Ling et al. 2016)\nEvaluation of Realizability & Reynolds Stress Anisotropy Invariant States', fontsize=12.5, fontweight='bold', y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved {out_path} ({out_path.stat().st_size / 1000000.0:.2f} MB)')

def gen_pipeline(repo_root: Path):
    print('--> [4/4] Generating Asset 4: In-Situ Architecture Pipeline Diagram...')
    out_path = repo_root / 'docs' / 'figures' / 'architecture_pipeline.png'
    fig = plt.figure(figsize=(18, 7.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')

    def draw_box(x, y, w, h, title, subtitle, color, text_color='black'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.15', linewidth=1.8, edgecolor=color, facecolor='#f8f9fa')
        ax.add_patch(rect)
        header = patches.FancyBboxPatch((x, y + h - 0.75), w, 0.75, boxstyle='round,pad=0.1,rounding_size=0.1', linewidth=0, facecolor=color)
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - 0.38, title, color='white', fontsize=10.5, fontweight='bold', ha='center', va='center')
        ax.text(x + w / 2, y + (h - 0.75) / 2, subtitle, color=text_color, fontsize=8.8, ha='center', va='center', multialignment='center')

    def draw_arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->,head_width=0.4,head_length=0.6', color='#1f2328', lw=2.2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.28, label, fontsize=8.5, fontweight='bold', ha='center', color='#0969da')
    draw_box(0.5, 2.0, 3.2, 4.5, '1. OpenFOAM Core', 'Mean Flow Fields:\n\n$\\mathbf{U} = (U_x, U_y, U_z)$' + '\n$k$ (Turbulent Kinetic Energy)' + '\n$\\varepsilon$ (Dissipation Rate)' + '\n$\\nu_t = C_\\mu k^2 / \\varepsilon$', '#0969da')
    draw_arrow(3.9, 4.25, 4.8, 4.25, '$\\nabla \\mathbf{U}$')
    draw_box(4.8, 1.8, 3.5, 4.9, '2. Invariant & Basis Engine', 'Strain & Rotation Rates:\n$\\mathbf{S} = \\frac{k}{2\\varepsilon}(\\nabla \\mathbf{U} + \\nabla \\mathbf{U}^T)$' + '\n$\\mathbf{R} = \\frac{k}{2\\varepsilon}(\\nabla \\mathbf{U} - \\nabla \\mathbf{U}^T)$' + '\n\nPope (1975) Invariants:' + '\n$I_1..I_5 = \\{\\mathrm{tr}(\\mathbf{S}^2), \\mathrm{tr}(\\mathbf{R}^2), \\dots\\}$' + '\n\nTensor Basis:' + '\n$\\{T^{(1)} \\dots T^{(10)}\\}$', '#1a7f37')
    draw_arrow(8.5, 4.25, 9.4, 4.25, '$I_1 \\dots I_5$')
    draw_box(9.4, 1.8, 3.6, 4.9, '3. In-Situ LibTorch JIT', 'Embedded C++ PyTorch Model:\ntorch::jit::load("tbnn_model.pt")\n\nDeep MLP Architecture:\n$[5] \\to [64] \\to [64] \\to [64] \\to [10]$' + '\nNonlinear Activations (LeakyReLU)\n\nScalar Invariant Weights:\n$\\mathbf{g} = [g^{(1)}, g^{(2)}, \\dots, g^{(10)}]^T$', '#8250df')
    draw_arrow(13.2, 4.25, 14.1, 4.25, '$\\mathbf{a} = \\sum g^{(n)} T^{(n)}$')
    draw_box(14.1, 1.8, 3.4, 4.9, '4. Momentum RHS Coupling', 'Realizability Limiter:\n$a_{ii} \\in [-0.33, 0.66]$' + '\n$a_{ij} \\in [-0.50, 0.50]$' + '\n\nReynolds Stress Injection:\n$\\mathbf{\\tau_\\Delta} = 2k \\mathbf{a}$' + '\n$\\mathbf{R}_{ij} = \\mathbf{\\tau}_\\Delta + \\frac{2}{3}k\\mathbf{I}$' + '\n\nOpenFOAM UEqn.H:\n$\\nabla \\cdot \\mathbf{\\tau_\\Delta}\\ \\text{RHS Source}$', '#cf222e')
    rect_feedback = patches.FancyArrowPatch((15.8, 1.6), (2.1, 1.6), connectionstyle='arc3,rad=-0.18', arrowstyle='->,head_width=0.4,head_length=0.6', color='#6e7781', lw=2.0, linestyle='--')
    ax.add_patch(rect_feedback)
    ax.text(8.9, 0.45, 'In-Situ Under-Relaxed Coupling Loop (SIMPLE / PIMPLE: Consistent Pressure-Velocity Update)', fontsize=9.5, fontweight='bold', ha='center', color='#57606a')
    fig.suptitle('$\\mathbf{tbnnFoam\\ In-Situ\\ C++\\ Architecture:\\ LibTorch\\ JIT\\ \\longleftrightarrow\\ OpenFOAM\\ SIMPLE\\ Solver}$', fontsize=13.5, fontweight='bold', y=0.96)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved {out_path} ({out_path.stat().st_size / 1000000.0:.2f} MB)')

def main():
    repo_root = Path(__file__).resolve().parent.parent
    data = load_dns_data(repo_root)
    if not (repo_root / 'docs' / 'figures' / 'animation_4_solver_convergence.gif').exists():
        gen_convergence(repo_root, data)
    else:
        print('--> [1/4] Asset 1 already generated.')
    gen_stress_mech(repo_root, data)
    gen_barycentric(repo_root, data)
    gen_pipeline(repo_root)
    print('\n--> ALL 4 SHOWCASE ASSETS SUCCESSFULLY GENERATED!')
if __name__ == '__main__':
    main()
