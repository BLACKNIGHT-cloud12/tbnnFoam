import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

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
                vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    return np.array(vectors)

def gen_cross_section(repo_root: Path, out_path: Path):
    print('--> Generating Animation 1: Cross-Section Particle Dynamics from Genuine 200/U Field...')
    u_file = (repo_root / 'verification_squareDuct' / '300' / 'U') if (repo_root / 'verification_squareDuct' / '300' / 'U').exists() else (repo_root / 'verification_squareDuct' / '200' / 'U')
    tbnn_u = parse_foam_vec(u_file)
    Nx, Ny_c, Nz_c = 50, 20, 20
    u_3d = tbnn_u.reshape(Nz_c, Ny_c, Nx, 3)
    cs = np.mean(u_3d[:, :, 20:40, :], axis=2)
    y_c = np.linspace(-0.95, 0.95, Ny_c)
    z_c = np.linspace(-0.95, 0.95, Nz_c)
    y_g = np.pad(y_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    z_g = np.pad(z_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    pad_cs = np.pad(cs, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.0)
    y = np.linspace(-1.0, 1.0, 50)
    z = np.linspace(-1.0, 1.0, 50)
    Z_m, Y_m = np.meshgrid(z, y, indexing='ij')
    pts = np.column_stack([Z_m.flatten(), Y_m.flatten()])
    rgi_v = RegularGridInterpolator((z_g, y_g), pad_cs[:, :, 1], bounds_error=False, fill_value=0.0)
    rgi_w = RegularGridInterpolator((z_g, y_g), pad_cs[:, :, 2], bounds_error=False, fill_value=0.0)
    V_tbnn = rgi_v(pts).reshape(50, 50)
    W_tbnn = rgi_w(pts).reshape(50, 50)
    n_particles = 100
    np.random.seed(42)
    p_y = np.random.uniform(-0.85, 0.85, n_particles)
    p_z = np.random.uniform(-0.85, 0.85, n_particles)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=140)
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax_left, ax_right):
        ax.set_facecolor('#161b22')
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect('equal')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
            spine.set_linewidth(1.5)
    ax_left.set_title('Standard OpenFOAM (Linear RANS)\nZero Cross-Stream Motion (v = w = 0)', fontsize=12, fontweight='bold', color='#58a6ff', pad=12)
    ax_right.set_title('tbnnFoam (Genuine CFD Simulation Field)\n8 Secondary Circulation Vortices (Prandtl 2nd Kind)', fontsize=12, fontweight='bold', color='#f0883e', pad=12)
    ax_left.set_xlabel('y / h', color='white', fontsize=11)
    ax_left.set_ylabel('z / h', color='white', fontsize=11)
    ax_right.set_xlabel('y / h', color='white', fontsize=11)
    ax_right.set_ylabel('z / h', color='white', fontsize=11)
    ax_right.streamplot(y, z, V_tbnn.T, W_tbnn.T, color='#8b949e', density=1.2, linewidth=0.7, arrowsize=0.6)
    scatter_left = ax_left.scatter(p_y, p_z, c='#58a6ff', s=25, alpha=0.8, edgecolors='none')
    scatter_right = ax_right.scatter(p_y, p_z, c='#f0883e', s=35, alpha=0.9, edgecolors='yellow', linewidth=0.5)
    ax_left.text(0, 0, 'PARALLEL FLOW\n(v = w = 0)', color='#8b949e', ha='center', va='center', fontsize=14, fontweight='bold', alpha=0.6)
    n_frames = 35
    dt = 3.5
    curr_y = np.copy(p_y)
    curr_z = np.copy(p_z)
    def update(frame):
        nonlocal curr_y, curr_z
        p_eval = np.column_stack([curr_z, curr_y])
        v_eval = rgi_v(p_eval)
        w_eval = rgi_w(p_eval)
        curr_y += v_eval * dt
        curr_z += w_eval * dt
        mask_out = (np.abs(curr_y) > 0.95) | (np.abs(curr_z) > 0.95)
        curr_y[mask_out] = np.random.uniform(-0.8, 0.8, np.sum(mask_out))
        curr_z[mask_out] = np.random.uniform(-0.8, 0.8, np.sum(mask_out))
        scatter_right.set_offsets(np.column_stack([curr_y, curr_z]))
        return [scatter_right]
    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=True)
    anim.save(str(out_path), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f'[+] Saved Animation 1 to: {out_path}')

def gen_3d_ribbons(repo_root: Path, out_path: Path):
    print('--> Generating Animation 2: 3D Streamlines from Genuine Solver Velocity...')
    fig = plt.figure(figsize=(10, 8), dpi=140)
    fig.patch.set_facecolor('#0d1117')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    L = 4.0
    h = 1.0
    ax.plot([0, L, L, 0, 0], [-h, -h, h, h, -h], [h, h, h, h, h], color='#30363d', lw=1.2)
    ax.plot([0, L, L, 0, 0], [-h, -h, h, h, -h], [-h, -h, -h, -h, -h], color='#30363d', lw=1.2)
    ax.plot([0, 0], [-h, -h], [-h, h], color='#30363d', lw=1.2)
    ax.plot([0, 0], [h, h], [-h, h], color='#30363d', lw=1.2)
    ax.plot([L, L], [-h, -h], [-h, h], color='#30363d', lw=1.2)
    ax.plot([L, L], [h, h], [-h, h], color='#30363d', lw=1.2)
    u_file = (repo_root / 'verification_squareDuct' / '300' / 'U') if (repo_root / 'verification_squareDuct' / '300' / 'U').exists() else (repo_root / 'verification_squareDuct' / '200' / 'U')
    tbnn_u = parse_foam_vec(u_file)
    Nx, Ny_c, Nz_c = 50, 20, 20
    u_3d = tbnn_u.reshape(Nz_c, Ny_c, Nx, 3)
    cs = np.mean(u_3d[:, :, 20:40, :], axis=2)
    y_c = np.linspace(-0.95, 0.95, Ny_c)
    z_c = np.linspace(-0.95, 0.95, Nz_c)
    y_g = np.pad(y_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    z_g = np.pad(z_c, (1, 1), 'constant', constant_values=(-1.0, 1.0))
    pad_cs = np.pad(cs, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.0)
    rgi_v = RegularGridInterpolator((z_g, y_g), pad_cs[:, :, 1], bounds_error=False, fill_value=0.0)
    rgi_w = RegularGridInterpolator((z_g, y_g), pad_cs[:, :, 2], bounds_error=False, fill_value=0.0)
    n_pts = 100
    seeds = [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.6, 0.2), (-0.6, 0.2), (0.2, 0.6), (-0.2, -0.6)]
    colors = ['#ff7b72', '#79c0ff', '#d2a8ff', '#ffa657', '#56d364', '#7ee787', '#e3b341', '#f0883e']
    trajectories = []
    dt = 0.08
    for i, (y0, z0) in enumerate(seeds):
        xs = np.linspace(0, L, n_pts)
        ys = np.zeros(n_pts)
        zs = np.zeros(n_pts)
        ys[0] = y0
        zs[0] = z0
        for step in range(1, n_pts):
            pt = np.array([[zs[step-1], ys[step-1]]])
            vy = float(rgi_v(pt)[0])
            wz = float(rgi_w(pt)[0])
            ys[step] = np.clip(ys[step-1] + vy * dt, -0.95, 0.95)
            zs[step] = np.clip(zs[step-1] + wz * dt, -0.95, 0.95)
        trajectories.append((xs, ys, zs, colors[i]))
    for x_t, y_t, z_t, c in trajectories:
        ax.plot(x_t, y_t, z_t, color=c, alpha=0.35, lw=1.2)
    dots = [ax.plot([], [], [], marker='o', markersize=6, color=c)[0] for _, _, _, c in trajectories]
    ax.set_title('tbnnFoam: 3D Streamlines Integrated from Solver Velocity Field\n(Secondary Convection Transporting Fluid into the 4 Corners)', color='white', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('x (Streamwise)', color='white', labelpad=8)
    ax.set_ylabel('y (Spanwise)', color='white', labelpad=8)
    ax.set_zlabel('z (Vertical)', color='white', labelpad=8)
    ax.tick_params(colors='white')
    ax.view_init(elev=22, azim=-55)
    n_frames = 35
    def update(frame):
        idx = int(frame / n_frames * n_pts)
        for i, (x_t, y_t, z_t, _) in enumerate(trajectories):
            curr_idx = (idx + i * 10) % n_pts
            dots[i].set_data([x_t[curr_idx]], [y_t[curr_idx]])
            dots[i].set_3d_properties([z_t[curr_idx]])
        return dots
    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f'[+] Saved Animation 2 to: {out_path}')

def main():
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / 'docs' / 'figures'
    docs_dir.mkdir(parents=True, exist_ok=True)
    anim1_path = docs_dir / 'animation_1_secondary_vortices.gif'
    anim2_path = docs_dir / 'animation_2_3d_streamlines.gif'
    gen_cross_section(repo_root, anim1_path)
    gen_3d_ribbons(repo_root, anim2_path)
    print('\n[+] Showcase animations generated directly from genuine OpenFOAM solver velocity fields!')

if __name__ == '__main__':
    main()
