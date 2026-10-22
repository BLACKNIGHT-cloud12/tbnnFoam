import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add repo root to path for vortex detector
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from scripts.validation.vortex_detector import detect_vortex_cores

def parse_foam_field(file_path):
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
        elif len(parts) == 6:
            try:
                vectors.append([float(p) for p in parts])
            except ValueError:
                continue
        elif len(parts) == 1:
            try:
                vectors.append(float(parts[0]))
            except ValueError:
                continue
    return np.array(vectors)

def main():
    case_dir = Path(__file__).resolve().parent
    time_dirs = [d for d in os.listdir(case_dir) if d.isdigit() and int(d) > 0]
    if not time_dirs:
        print("[!] No time directories found yet.")
        return
    latest_time = sorted(time_dirs, key=lambda x: int(x))[-1]
    print(f"[+] Analyzing case at Re_b = 17,800, latest converged step: Time = {latest_time}")

    u_path = case_dir / latest_time / 'U'
    U_raw = parse_foam_field(u_path)
    if U_raw is None:
        print(f"[!] Failed to parse {u_path}")
        return

    # Geometry & grid parameters: 50 cells in X, 40 in Y, 40 in Z
    Nx, Ny, Nz = 50, 40, 40
    h = 0.05  # half width [m]
    Dh = 0.10 # hydraulic diameter [m]
    Ub = 1.0  # bulk velocity [m/s]
    nu = 5.617978e-6 # m^2/s -> Re_b = 17,800
    Re_b = Ub * Dh / nu
    Re_tau_lit = 519.0 # Pirozzoli et al. (2018) DNS reference at Re_b = 17,800

    # OpenFOAM block ordering: Z fastest/outer, Y middle, X inner (incrementing along X)
    # Shape: (Nz, Ny, Nx, 3)
    U_3d = U_raw.reshape(Nz, Ny, Nx, 3)

    # Sample developed cross-section downstream (x/Dh ~ 5-7, indices 25 to 45)
    U_cs = np.mean(U_3d[:, :, 25:45, :], axis=2)
    # Components: 0 -> Ux, 1 -> Uy, 2 -> Uz
    Ux = U_cs[:, :, 0]
    Uy = U_cs[:, :, 1]
    Uz = U_cs[:, :, 2]

    # Non-dimensional coordinates: y/h in [-1, 1], z/h in [-1, 1]
    y_norm = np.linspace(-1.0 + 1.0/Ny, 1.0 - 1.0/Ny, Ny)
    z_norm = np.linspace(-1.0 + 1.0/Nz, 1.0 - 1.0/Nz, Nz)
    Y_grid, Z_grid = np.meshgrid(y_norm, z_norm)

    # Non-dimensional velocities
    u_star = Ux / Ub
    v_star = Uy / Ub
    w_star = Uz / Ub
    v_sec = np.sqrt(v_star**2 + w_star**2)

    # Non-dimensional axial vorticity: omega_x / (Ub / h) = d(w*)/d(y/h) - d(v*)/d(z/h)
    dy_norm = y_norm[1] - y_norm[0]
    dz_norm = z_norm[1] - z_norm[0]
    dw_dy = np.gradient(w_star, dy_norm, axis=1)
    dv_dz = np.gradient(v_star, dz_norm, axis=0)
    omega_x_star = dw_dy - dv_dz

    # Run automated vortex detector for the primary 8 corner vortices
    detected_cores = detect_vortex_cores(
        omega_x_star,
        y_norm,
        z_norm,
        min_separation=0.25,
        min_strength=0.15,
        min_area=4,
        wall_margin=0.08
    )

    peak_vsec_pct = np.max(v_sec) * 100.0

    print("=" * 70)
    print("      TBNN-OpenFOAM High-Reynolds Number Validation (Re_b = 17,800)")
    print("=" * 70)
    print(f"Bulk Reynolds Number (Re_b)      : {Re_b:.1f}")
    print(f"Friction Reynolds Number (Re_tau): ~{Re_tau_lit:.0f} (Pirozzoli et al. 2018)")
    print(f"Peak Secondary Speed (V_sec/U_b) : {peak_vsec_pct:.2f}%  (DNS benchmark: ~2.1 - 2.3%)")
    print(f"Vortex Cores Detected            : {len(detected_cores)} / 8 canonical cores")
    print("Detected Core Coordinates:")
    for idx, c in enumerate(detected_cores):
        print(f"  Core #{idx+1:02d}: (y/h={c['y']:.3f}, z/h={c['z']:.3f}), Type: {c['type']}, Strength: {c['strength']:.3f}")
    print("=" * 70)

    # Plot 4-panel publication showcase
    fig = plt.figure(figsize=(15, 13), dpi=250)
    gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.25)

    # (a) Streamwise Velocity Contours U/Ub with Streamlines
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.contourf(Y_grid, Z_grid, u_star, levels=30, cmap='viridis')
    cb1 = plt.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label(r'$U / U_b$', fontsize=12)
    # Streamlines of secondary flow
    ax1.streamplot(y_norm, z_norm, v_star, w_star, color='white', density=1.2, linewidth=0.7, arrowsize=0.8)
    ax1.set_title(r'(a) Streamwise Velocity $U / U_b$ & Secondary Streamlines', fontsize=13, fontweight='bold')
    ax1.set_xlabel(r'$y / h$', fontsize=11)
    ax1.set_ylabel(r'$z / h$', fontsize=11)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal')

    # (b) In-Plane Secondary Flow Velocity Vectors
    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = ax2.contourf(Y_grid, Z_grid, v_sec * 100.0, levels=30, cmap='inferno')
    cb2 = plt.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label(r'$V_{\mathrm{sec}} / U_b$ (%)', fontsize=12)
    # Quiver plot (subsampled for clarity)
    step = 2
    ax2.quiver(Y_grid[::step, ::step], Z_grid[::step, ::step],
               v_star[::step, ::step], w_star[::step, ::step],
               color='cyan', scale=0.6, width=0.003, headwidth=4)
    ax2.set_title(r'(b) Secondary Velocity Magnitude & Vectors ($V_{\mathrm{sec}}/U_b$)', fontsize=13, fontweight='bold')
    ax2.set_xlabel(r'$y / h$', fontsize=11)
    ax2.set_ylabel(r'$z / h$', fontsize=11)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_aspect('equal')

    # (c) Axial Vorticity Field omega_x^*
    ax3 = fig.add_subplot(gs[1, 0])
    vort_lim = max(0.25, np.percentile(np.abs(omega_x_star), 98))
    cf3 = ax3.contourf(Y_grid, Z_grid, omega_x_star, levels=np.linspace(-vort_lim, vort_lim, 31),
                       cmap='seismic', extend='both')
    cb3 = plt.colorbar(cf3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label(r'$\omega_x / (U_b / h)$', fontsize=12)
    ax3.set_title(r'(c) Non-Dimensional Axial Vorticity $\omega_x^*$', fontsize=13, fontweight='bold')
    ax3.set_xlabel(r'$y / h$', fontsize=11)
    ax3.set_ylabel(r'$z / h$', fontsize=11)
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_aspect('equal')

    # (d) Detected 8 Secondary Vortex Cores Overlaid on Vorticity
    ax4 = fig.add_subplot(gs[1, 1])
    cf4 = ax4.contourf(Y_grid, Z_grid, np.abs(omega_x_star), levels=25, cmap='magma')
    cb4 = plt.colorbar(cf4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label(r'$|\omega_x^*|$', fontsize=12)
    for c in detected_cores:
        marker = 'o' if c['type'] == 'acw' else 's'
        color = 'cyan' if c['type'] == 'acw' else 'lime'
        ax4.scatter(c['y'], c['z'], s=90, c=color, edgecolors='black', linewidth=1.2,
                    marker=marker, zorder=5)
    ax4.scatter([], [], c='cyan', edgecolors='black', marker='o', label='Anti-Clockwise Core (ACW)')
    ax4.scatter([], [], c='lime', edgecolors='black', marker='s', label='Clockwise Core (CW)')
    ax4.legend(loc='lower center', framealpha=0.9, fontsize=10)
    ax4.set_title(f'(d) Automated Vortex Detection ({len(detected_cores)}/8 Cores Resolved)', fontsize=13, fontweight='bold')
    ax4.set_xlabel(r'$y / h$', fontsize=11)
    ax4.set_ylabel(r'$z / h$', fontsize=11)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-1, 1)
    ax4.set_aspect('equal')

    fig.suptitle(r'$\mathbf{tbnnFoam}$ In-Situ RANS Simulation at High Reynolds Number ($Re_b = 17{,}800$, $Re_\tau \approx 519$)',
                 fontsize=16, fontweight='bold', y=0.98)

    save_path = case_dir / 'tbnn_Re17800_showcase.png'
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved showcase figure to: {save_path}")

if __name__ == '__main__':
    main()
