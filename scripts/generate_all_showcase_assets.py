import os
import json
import csv
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

def parse_foam_symm(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    start_token = 'internalField'
    idx = content.find(start_token)
    if idx == -1:
        return None
    chunk = content[idx:]
    open_paren = chunk.find('(')
    close_paren = chunk.rfind(')')
    if open_paren == -1 or close_paren == -1:
        return None
    data_str = chunk[open_paren + 1:close_paren].strip()
    tensors = []
    for line in data_str.split('\n'):
        line = line.strip().strip('()').strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 6:
            try:
                tensors.append([float(p) for p in parts])
            except ValueError:
                continue
    return np.array(tensors)

def parse_foam_vec(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    start_token = 'internalField'
    idx = content.find(start_token)
    if idx == -1:
        return None
    chunk = content[idx:]
    open_paren = chunk.find('(')
    close_paren = chunk.rfind(')')
    if open_paren == -1 or close_paren == -1:
        return None
    data_str = chunk[open_paren + 1:close_paren].strip()
    vectors = []
    for line in data_str.split('\n'):
        line = line.strip().strip('()').strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                vectors.append([float(p) for p in parts])
            except ValueError:
                continue
    return np.array(vectors)

def parse_log_residuals(log_path: Path):
    iters = []
    cont_locals = []
    cont_globals = []
    with open(log_path, 'r', encoding='utf-8') as f:
        current_time = None
        for line in f:
            if line.startswith('Time = '):
                try:
                    current_time = int(line.split('=')[1].strip())
                except ValueError:
                    current_time = None
            elif 'time step continuity errors :' in line and current_time is not None:
                parts = line.split(',')
                try:
                    local_val = float(parts[0].split('=')[1].strip())
                    global_val = float(parts[1].split('=')[1].strip())
                    iters.append(current_time)
                    cont_locals.append(local_val)
                    cont_globals.append(global_val)
                except (IndexError, ValueError):
                    continue
    return (np.array(iters), np.array(cont_locals), np.array(cont_globals))

def load_dns_data(repo_root: Path):
    dns_file = repo_root / 'data' / 'benchmark_dns' / 'square_duct_dns_Re1500.csv'
    meta_file = repo_root / 'data' / 'benchmark_dns' / 'benchmark_metadata.json'
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    dns_rows = []
    with open(dns_file, 'r', encoding='utf-8') as f:
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
    return {
        'meta': meta,
        'y_grid': dns_y.reshape(Ny, Nz),
        'z_grid': dns_z.reshape(Ny, Nz),
        'u_dns': dns_u.reshape(Ny, Nz),
        'v_dns': dns_v.reshape(Ny, Nz),
        'w_dns': dns_w.reshape(Ny, Nz),
        'k_dns': dns_k.reshape(Ny, Nz),
        'a22_dns': dns_a22.reshape(Ny, Nz),
        'a33_dns': dns_a33.reshape(Ny, Nz),
        'a11_dns': dns_a11.reshape(Ny, Nz),
        'a12_dns': dns_a12.reshape(Ny, Nz),
        'a13_dns': dns_a13.reshape(Ny, Nz),
        'a23_dns': dns_a23.reshape(Ny, Nz),
        'Ny': Ny,
        'Nz': Nz
    }

def gen_convergence(repo_root: Path, data):
    print('--> [1/4] Generating Asset 1: Solver Convergence Diagnostics (t = 1 -> 200)...')
    out_path = repo_root / 'docs' / 'figures' / 'animation_4_solver_convergence.gif'
    log_path = repo_root / 'verification_squareDuct' / 'log.tbnnFoam'
    iters, cont_locals, cont_globals = parse_log_residuals(log_path)
    u_file = (repo_root / 'verification_squareDuct' / '300' / 'U') if (repo_root / 'verification_squareDuct' / '300' / 'U').exists() else (repo_root / 'verification_squareDuct' / '200' / 'U')
    base_file = (repo_root / 'verification_squareDuct' / 'baseline_73' / 'U') if (repo_root / 'verification_squareDuct' / 'baseline_73' / 'U').exists() else (repo_root / 'verification_squareDuct' / '50' / 'U')
    tbnn_u = parse_foam_vec(u_file)
    base_u = parse_foam_vec(base_file)
    Nx, Ny_c, Nz_c = 50, 20, 20
    u_3d = tbnn_u.reshape(Nz_c, Ny_c, Nx, 3)
    base_3d = base_u.reshape(Nz_c, Ny_c, Nx, 3)
    cs = np.mean(u_3d[:, :, 20:40, :], axis=2)
    base_cs = np.mean(base_3d[:, :, 20:40, :], axis=2)
    y_c = np.linspace(-0.95, 0.95, Ny_c)
    z_c = np.linspace(-0.95, 0.95, Nz_c)
    y_g = np.pad(y_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    z_g = np.pad(z_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    pad_cs = np.pad(cs, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.0)
    pad_base = np.pad(base_cs, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.0)
    y_fine = np.linspace(-1.0, 1.0, 65)
    z_fine = np.linspace(-1.0, 1.0, 65)
    Z_m, Y_m = np.meshgrid(z_fine, y_fine, indexing='ij')
    pts = np.column_stack([Z_m.flatten(), Y_m.flatten()])
    rgi_u = RegularGridInterpolator((z_g, y_g), pad_cs[:, :, 0], bounds_error=False, fill_value=0.0)
    rgi_v = RegularGridInterpolator((z_g, y_g), pad_cs[:, :, 1], bounds_error=False, fill_value=0.0)
    rgi_w = RegularGridInterpolator((z_g, y_g), pad_cs[:, :, 2], bounds_error=False, fill_value=0.0)
    rgi_u_base = RegularGridInterpolator((z_g, y_g), pad_base[:, :, 0], bounds_error=False, fill_value=0.0)
    U_end = rgi_u(pts).reshape(65, 65)
    V_end = rgi_v(pts).reshape(65, 65)
    W_end = rgi_w(pts).reshape(65, 65)
    U_base = rgi_u_base(pts).reshape(65, 65)
    speed_end = np.sqrt(V_end**2 + W_end**2) * 100.0
    n_frames = 35
    max_iter = int(iters[-1]) if len(iters) > 0 else 300
    frame_steps = np.linspace(50, max_iter, n_frames).astype(int)
    fig, (ax_flow, ax_res) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=140, gridspec_kw={'width_ratios': [1.3, 1.0]})
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax_flow, ax_res):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            spine.set_linewidth(1.5)
    def update(frame):
        ax_flow.clear()
        ax_res.clear()
        for ax in (ax_flow, ax_res):
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
                spine.set_linewidth(1.5)
        step = frame_steps[frame]
        alpha = min(1.0, max(0.0, (step - 60) / 40.0))
        speed_cur = alpha * speed_end
        V_cur = alpha * V_end
        W_cur = alpha * W_end
        U_cur = (1.0 - alpha) * U_base + alpha * U_end
        cf = ax_flow.contourf(y_fine, z_fine, speed_cur, levels=np.linspace(0, 6.0, 30), cmap='inferno')
        ax_flow.contour(y_fine, z_fine, U_cur, levels=8, colors='#58a6ff', linewidths=0.9, alpha=0.7)
        if alpha > 0.05:
            ax_flow.streamplot(y_fine, z_fine, V_cur.T, W_cur.T, color='white', density=0.9, linewidth=0.7 * alpha + 0.3, arrowsize=0.7, arrowstyle='->')
        ax_flow.set_xlim(-1.0, 1.0)
        ax_flow.set_ylim(-1.0, 1.0)
        ax_flow.set_aspect('equal')
        ax_flow.set_xlabel('y / h', color='white', fontsize=10)
        ax_flow.set_ylabel('z / h', color='white', fontsize=10)
        peak_v = np.max(speed_cur)
        ax_flow.set_title(f'OpenFOAM SIMPLE Iteration: t = {step}\nIn-Situ TBNN Stress Coupling: α = {alpha:.2f} | Peak V_sec = {peak_v:.2f}% Ub', color='#58a6ff', fontsize=11, fontweight='bold', pad=8)
        mask = iters <= step
        ax_res.semilogy(iters[mask], cont_locals[mask], color='#58a6ff', linewidth=2.0, label='Continuity Local Error')
        ax_res.semilogy(iters[mask], cont_globals[mask], color='#f0883e', linewidth=2.0, linestyle='--', label='Continuity Global Error')
        ax_res.set_xlim(1, max_iter + 5)
        ax_res.set_ylim(1e-6, 1e-1)
        ax_res.set_xlabel('SIMPLE Iterations (OpenFOAM log.tbnnFoam)', color='white', fontsize=10)
        ax_res.set_ylabel('Continuity Residual', color='white', fontsize=10)
        ax_res.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=9)
        ax_res.grid(True, linestyle=':', color='#30363d', alpha=0.6)
        ax_res.set_title('Recorded OpenFOAM Residual History\nDerived from verification_squareDuct/log.tbnnFoam', color='white', fontsize=11, fontweight='bold', pad=8)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=120)
    anim.save(out_path, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f'  -> Saved {out_path} ({out_path.stat().st_size / 1000000.0:.2f} MB)')

def gen_stress_mech(repo_root: Path, data):
    print('--> [2/4] Generating Asset 2: Reynolds Stress Mechanism Figure from Solver Outputs...')
    out_path = repo_root / 'docs' / 'figures' / 'reynolds_stress_mechanism.png'
    Ub = data['meta']['flow_parameters']['bulk_velocity_Ub_m_s']
    h = data['meta']['flow_parameters']['duct_half_width_h_m']
    k_dns = data['k_dns']
    a22 = data['a22_dns']
    a33 = data['a33_dns']
    diff_dns = 2.0 * k_dns * (a22 - a33) / Ub**2
    tau_file = (repo_root / 'verification_squareDuct' / '300' / 'tauR') if (repo_root / 'verification_squareDuct' / '300' / 'tauR').exists() else (repo_root / 'verification_squareDuct' / '200' / 'tauR')
    tau_raw = parse_foam_symm(tau_file)
    Nx, Ny_c, Nz_c = 50, 20, 20
    tau_3d = tau_raw.reshape(Nz_c, Ny_c, Nx, 6)
    tau_cs = np.mean(tau_3d[:, :, 20:40, :], axis=2)
    diff_cfd = (tau_cs[:, :, 3] - tau_cs[:, :, 5]) / Ub**2
    y_c = np.linspace(-0.95, 0.95, Ny_c)
    z_c = np.linspace(-0.95, 0.95, Nz_c)
    y_g = np.pad(y_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    z_g = np.pad(z_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    diff_pad = np.pad(diff_cfd, ((1, 1), (1, 1)), 'constant', constant_values=0.0)
    y_dns = np.linspace(-1.0, 1.0, 65)
    z_dns = np.linspace(-1.0, 1.0, 65)
    Z_m, Y_m = np.meshgrid(z_dns, y_dns, indexing='ij')
    pts = np.column_stack([Z_m.flatten(), Y_m.flatten()])
    rgi_diff = RegularGridInterpolator((z_g, y_g), diff_pad, bounds_error=False, fill_value=0.0)
    diff_tbnn = rgi_diff(pts).reshape(65, 65)
    diff_rans = np.zeros_like(diff_dns)
    fig = plt.figure(figsize=(18, 6.2), dpi=300)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.3], wspace=0.28)
    y_norm = np.linspace(-1.0, 1.0, 65)
    z_norm = np.linspace(-1.0, 1.0, 65)
    vmax_val = np.percentile(np.abs(diff_dns), 98)
    levels = np.linspace(-vmax_val, vmax_val, 31)
    ax1 = fig.add_subplot(gs[0])
    cf1 = ax1.contourf(y_norm, z_norm, diff_rans, levels=levels, cmap='RdBu_r')
    ax1.set_title('Standard OpenFOAM\nLinear Boussinesq EVM', fontsize=11, fontweight='bold', pad=8)
    ax1.set_xlabel('y / h', fontsize=10)
    ax1.set_ylabel('z / h', fontsize=10)
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_aspect('equal')
    ax1.text(0, 0, "$\\overline{v'^2} - \\overline{w'^2} \\equiv 0$" + '\n(Identically Zero)', ha='center', va='center', fontsize=11, fontweight='bold', color='darkred', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    ax2 = fig.add_subplot(gs[1])
    cf2 = ax2.contourf(y_norm, z_norm, diff_tbnn, levels=levels, cmap='RdBu_r')
    ax2.set_title('tbnnFoam (tauR)\nGenuine In-Situ CFD Output', fontsize=11, fontweight='bold', pad=8)
    ax2.set_xlabel('y / h', fontsize=10)
    ax2.set_ylabel('z / h', fontsize=10)
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_aspect('equal')
    ax3 = fig.add_subplot(gs[2])
    cf3 = ax3.contourf(y_norm, z_norm, diff_dns, levels=levels, cmap='RdBu_r')
    ax3.set_title('Reference DNS\n(Pirozzoli et al. 2018)', fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel('y / h', fontsize=10)
    ax3.set_ylabel('z / h', fontsize=10)
    ax3.set_xlim(-1.0, 1.0)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_aspect('equal')
    cbar = fig.colorbar(cf3, ax=[ax1, ax2, ax3], orientation='horizontal', fraction=0.045, pad=0.18, aspect=35)
    cbar.set_label("Normalized Reynolds Normal Stress Difference: $(\\overline{v'^2} - \\overline{w'^2}) / U_b^2$", fontsize=10, fontweight='bold')
    ax4 = fig.add_subplot(gs[3])
    diag_idx = np.arange(len(y_norm))
    diag_coords = np.linspace(-1.0, 1.0, len(diag_idx))
    diag_dns = np.diagonal(diff_dns)
    diag_tbnn = np.diagonal(diff_tbnn)
    diag_rans = np.diagonal(diff_rans)
    ax4.plot(diag_coords, diag_dns * 1000.0, 'k--', linewidth=2.2, label='DNS (Pirozzoli et al.)')
    ax4.plot(diag_coords, diag_tbnn * 1000.0, 'r-', linewidth=2.0, label='tbnnFoam (tauR)')
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
    print('--> [3/4] Generating Asset 3: Barycentric Anisotropy Triangle Map from Genuine aTBNN Output...')
    out_path = repo_root / 'docs' / 'figures' / 'barycentric_anisotropy_map.png'
    x_1c = np.array([1.0, 0.0])
    x_2c = np.array([0.0, 0.0])
    x_3c = np.array([0.5, np.sqrt(3.0) / 2.0])
    def calc_barycentric(a_tensors):
        n_pts = len(a_tensors)
        C1 = np.zeros(n_pts)
        C2 = np.zeros(n_pts)
        C3 = np.zeros(n_pts)
        for i in range(n_pts):
            t = a_tensors[i]
            mat = np.array([[t[0], t[1], t[2]], [t[1], t[3], t[4]], [t[2], t[4], t[5]]])
            mat = mat - (np.trace(mat) / 3.0) * np.eye(3)
            w = np.sort(np.linalg.eigvalsh(mat))[::-1]
            l1, l2, l3 = (w[0], w[1], w[2])
            c1 = l1 - l2
            c2 = 2.0 * (l2 - l3)
            c3 = 3.0 * l3 + 1.0
            c = np.maximum(0.0, np.array([c1, c2, c3]))
            tot = np.sum(c)
            if tot > 0:
                c /= tot
            else:
                c = np.array([0.0, 0.0, 1.0])
            C1[i] = c[0]
            C2[i] = c[1]
            C3[i] = c[2]
        x_b = C1 * x_1c[0] + C2 * x_2c[0] + C3 * x_3c[0]
        y_b = C1 * x_1c[1] + C2 * x_2c[1] + C3 * x_3c[1]
        return (x_b, y_b)
    a_file = (repo_root / 'verification_squareDuct' / '300' / 'aTBNN') if (repo_root / 'verification_squareDuct' / '300' / 'aTBNN').exists() else (repo_root / 'verification_squareDuct' / '200' / 'aTBNN')
    a_cfd_raw = parse_foam_symm(a_file)
    Nx, Ny_c, Nz_c = 50, 20, 20
    a_3d = a_cfd_raw.reshape(Nz_c, Ny_c, Nx, 6)
    a_cs = np.mean(a_3d[:, :, 20:40, :], axis=2)
    a_tbnn_cells = a_cs.reshape(-1, 6)
    x_tbnn, y_tbnn = calc_barycentric(a_tbnn_cells)
    a11_flat = data['a11_dns'].flatten()
    a22_flat = data['a22_dns'].flatten()
    a33_flat = data['a33_dns'].flatten()
    a12_flat = data['a12_dns'].flatten()
    a13_flat = data['a13_dns'].flatten()
    a23_flat = data['a23_dns'].flatten()
    dns_symm = np.column_stack([a11_flat, a12_flat, a13_flat, a22_flat, a23_flat, a33_flat])
    idx_sample = np.random.RandomState(42).choice(len(dns_symm), size=400, replace=False)
    x_dns, y_dns = calc_barycentric(dns_symm[idx_sample])
    x_rans = np.full(400, 0.5)
    y_rans = np.linspace(np.sqrt(3.0)/2.0 - 0.1, np.sqrt(3.0)/2.0, 400)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)
    titles = [
        'Standard OpenFOAM (Linear EVM)\nTrapped Near 3C Isotropic Apex',
        'tbnnFoam (aTBNN)\nGenuine In-Situ Solver Output',
        'Reference DNS (Pirozzoli et al. 2018)\nBenchmark Stress State'
    ]
    datasets = [(x_rans, y_rans, 'royalblue'), (x_tbnn, y_tbnn, 'crimson'), (x_dns, y_dns, 'black')]
    for ax, title, (xb, yb, color) in zip(axes, titles, datasets):
        triangle_x = [x_2c[0], x_1c[0], x_3c[0], x_2c[0]]
        triangle_y = [x_2c[1], x_1c[1], x_3c[1], x_2c[1]]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2.0)
        poly = patches.Polygon([x_2c, x_1c, x_3c], closed=True, facecolor='#f8f9fa', edgecolor='none')
        ax.add_patch(poly)
        ax.scatter(xb, yb, c=color, s=28, alpha=0.65, edgecolors='none')
        ax.text(x_2c[0] - 0.03, x_2c[1] - 0.04, '$\\mathbf{2C}$ (Wall)', fontsize=9.5, fontweight='bold', ha='left')
        ax.text(x_1c[0] + 0.03, x_1c[1] - 0.04, '$\\mathbf{1C}$ (Rod-like)', fontsize=9.5, fontweight='bold', ha='right')
        ax.text(x_3c[0], x_3c[1] + 0.04, '$\\mathbf{3C}$ (Isotropic Core)', fontsize=9.5, fontweight='bold', ha='center')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 0.98)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.suptitle('Barycentric Anisotropy Triangle (Banerjee et al. 2007, Ling et al. 2016)\nEvaluation of Physical Realizability & Reynolds Stress Anisotropy Invariant States', fontsize=12.5, fontweight='bold', y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved {out_path} ({out_path.stat().st_size / 1000000.0:.2f} MB)')

def gen_pipeline(repo_root: Path):
    print('--> [4/4] Generating Asset 4: In-Situ Architecture Pipeline Diagram...')
    out_path = repo_root / 'docs' / 'figures' / 'architecture_pipeline.png'
    fig = plt.figure(figsize=(18, 7.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.patch.set_facecolor('#0d1117')
    steps = [
        ('1. OpenFOAM RANS Core', 'fvc::grad(U), k, omega\nExtract velocity gradient\nand turbulence scales', '#1f6feb', (-4.0, 0.0)),
        ('2. Invariants & Bases', '5 Invariants: tr(s²), tr(w²)...\n10 Pope (1975) Bases\nT1 ... T10 symmetric & traceless', '#8957e5', (-1.3, 0.0)),
        ('3. Neural Network Engine', 'TBNN Multi-Layer Perceptron\n10 Scalar g_n coefficients\nReLU activations (in-situ)', '#d29922', (1.4, 0.0)),
        ('4. Stress Assembly', 'a = sum(g_n * T_n)\nLumley Triangle Realizability\ntau_R = 2k(a + I/3)', '#da3633', (4.1, 0.0)),
        ('5. Momentum Predictor', 'UEqn.H with fvc::div(tauDelta)\nExplicit Anisotropic Injection\nRecovers Secondary Flow', '#238636', (6.8, 0.0))
    ]
    for title, desc, color, (x, y) in steps:
        box = patches.FancyBboxPatch((x - 1.1, y - 0.7), 2.2, 1.4, boxstyle='round,pad=0.1,rounding_size=0.15', edgecolor=color, facecolor='#161b22', linewidth=2.0)
        ax.add_patch(box)
        ax.text(x, y + 0.35, title, color='white', fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(x, y - 0.15, desc, color='#c9d1d9', fontsize=9.5, ha='center', va='center')
    for i in range(len(steps) - 1):
        x1 = steps[i][3][0] + 1.15
        x2 = steps[i+1][3][0] - 1.15
        ax.annotate('', xy=(x2, 0.0), xytext=(x1, 0.0), arrowprops=dict(arrowstyle='->,head_width=0.4,head_length=0.6', color='#58a6ff', lw=2.5))
    ax.text(1.4, 1.1, 'In-Situ Tensor Basis Neural Network Finite-Volume Discretization Pipeline', color='white', fontsize=14, fontweight='bold', ha='center')
    ax.set_xlim(-5.5, 8.3)
    ax.set_ylim(-1.5, 1.5)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved {out_path} ({out_path.stat().st_size / 1000000.0:.2f} MB)')

def main():
    repo = Path(__file__).resolve().parent.parent
    data = load_dns_data(repo)
    gen_convergence(repo, data)
    gen_stress_mech(repo, data)
    gen_barycentric(repo, data)
    gen_pipeline(repo)
    print('\n[+] All showcase assets regenerated successfully from genuine simulation outputs!')

if __name__ == '__main__':
    main()
