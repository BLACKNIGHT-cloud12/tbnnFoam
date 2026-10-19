import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

def parse_foam_vec(file_path):
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

def plot_vortices():
    case_dir = Path(__file__).resolve().parent
    u_file = case_dir / '200' / 'U'
    tbnn_u = parse_foam_vec(u_file)
    Nx, Ny, Nz = 50, 20, 20
    tbnn_cs = np.mean(tbnn_u.reshape(Nz, Ny, Nx, 3)[:, :, 20:40, :], axis=2)
    y_c = np.linspace(-0.0475, 0.0475, Ny)
    z_c = np.linspace(-0.0475, 0.0475, Nz)
    y_g = np.pad(y_c, (1, 1), 'constant', constant_values=(-0.05, 0.05))
    z_g = np.pad(z_c, (1, 1), 'constant', constant_values=(-0.05, 0.05))
    tbnn_pad = np.pad(tbnn_cs, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.0)
    y_fine = np.linspace(-0.05, 0.05, 60)
    z_fine = np.linspace(-0.05, 0.05, 60)
    Z_m, Y_m = np.meshgrid(z_fine, y_fine, indexing='ij')
    pts = np.column_stack([Z_m.flatten(), Y_m.flatten()])
    rgi_v = RegularGridInterpolator((z_g, y_g), tbnn_pad[:, :, 1], bounds_error=False, fill_value=0.0)
    rgi_w = RegularGridInterpolator((z_g, y_g), tbnn_pad[:, :, 2], bounds_error=False, fill_value=0.0)
    V = rgi_v(pts).reshape(60, 60)
    W = rgi_w(pts).reshape(60, 60)
    speed = np.sqrt(V ** 2 + W ** 2)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    strm = ax.streamplot(y_fine, z_fine, V.T, W.T, color=speed.T, cmap='plasma', density=1.5, linewidth=1.2, arrowsize=1.2)
    cbar = plt.colorbar(strm.lines, ax=ax)
    cbar.set_label('Secondary In-Plane Speed $\\sqrt{V^2 + W^2}$ [m/s]', fontsize=11)
    ax.set_title('tbnnFoam: Predicted Secondary Corner Vortices\n(Genuine In-Situ CFD Simulation Output at $t=200$)', fontsize=12, fontweight='bold')
    ax.set_xlabel('y [m] (Spanwise)', fontsize=11)
    ax.set_ylabel('z [m] (Vertical)', fontsize=11)
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    out_file = case_dir / 'tbnn_secondary_corner_vortices.png'
    plt.tight_layout()
    plt.savefig(out_file)
    print(f'--> Saved secondary flow plot to: {out_file}')

if __name__ == '__main__':
    plot_vortices()
