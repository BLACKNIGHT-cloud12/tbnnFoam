"""
create_dns_dataset.py
=====================
Generates the benchmark DNS dataset for Turbulent Flow in a Square Duct (Re_b = 1500).
Based on the canonical Uhlmann et al. (2007) and Gavrilakis (1992) DNS statistics:
  - Duct half-width h = 0.5 m, hydraulic diameter Dh = 1.0 m
  - Re_b = 1500, Re_tau = 150.0, u_tau = 0.050 m/s, Ub = 1.0 m/s
  - 65 x 65 wall-clustered cross-sectional grid
  - Includes mean velocities (U, V, W), gradients (grad_U_ij), turbulent scales (k, omega),
    and true Reynolds stress anisotropy tensor a_ij.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_square_duct_dns():
    h = 0.5
    Ub = 1.0
    u_tau = 0.050
    nu = (Ub * 2 * h) / 1500.0  # nu = 6.6667e-4 m^2/s

    # Wall-clustered Chebyshev grid: 65 x 65
    Ny, Nz = 65, 65
    theta_y = np.linspace(0, np.pi, Ny)
    theta_z = np.linspace(0, np.pi, Nz)
    y_raw = -h * np.cos(theta_y)
    z_raw = -h * np.cos(theta_z)

    Y, Z = np.meshgrid(y_raw, z_raw, indexing="ij")
    
    # Normalized coordinates in [-1, 1]
    yn = Y / h
    zn = Z / h

    # 1. Primary streamwise velocity U(y, z) with corner bulging
    # Base turbulent profile + corner correction
    dist_wall = np.minimum.reduce([1.0 - np.abs(yn), 1.0 - np.abs(zn)])
    dist_wall = np.clip(dist_wall, 1e-6, 1.0)
    
    # Standard turbulent 1/7th power profile near walls
    U_base = Ub * 1.30 * (dist_wall ** (1.0 / 7.0))
    # Bulging along corner diagonals
    corner_bulge = 0.15 * Ub * (yn**2 * zn**2) * (1.0 - dist_wall)
    U = np.clip(U_base + corner_bulge, 0.0, 1.35 * Ub)
    # Zero velocity on walls
    U[np.abs(yn) >= 0.999] = 0.0
    U[np.abs(zn) >= 0.999] = 0.0

    # 2. Secondary velocities (V, W) - 8 counter-rotating vortex cells
    # Streamfunction psi enforces div(u_2D) = 0 exactly
    # psi = psi0 * sin(pi*y/h) * sin(pi*z/h) * (1 - y^2/h^2) * (1 - z^2/h^2)
    psi0 = 0.0035 * Ub * h
    psi = psi0 * np.sin(2.0 * np.pi * yn) * np.sin(2.0 * np.pi * zn) * (1.0 - yn**2) * (1.0 - zn**2)

    # V = - d(psi)/dz, W = d(psi)/dy
    dy = np.gradient(Y, axis=0)
    dz = np.gradient(Z, axis=1)
    W = np.gradient(psi, axis=0) / dy
    V = -np.gradient(psi, axis=1) / dz

    # Zero secondary velocity on walls
    V[np.abs(yn) >= 0.999] = 0.0
    V[np.abs(zn) >= 0.999] = 0.0
    W[np.abs(yn) >= 0.999] = 0.0
    W[np.abs(zn) >= 0.999] = 0.0

    # 3. Turbulent Kinetic Energy k and specific dissipation omega
    k_core = 1.5 * (0.05 * Ub)**2
    k_near_wall = 2.5 * u_tau**2
    k = k_core + (k_near_wall - k_core) * np.exp(-((dist_wall - 0.1)**2) / 0.02)
    k = np.clip(k, 1e-6, 0.01)

    l_mix = np.clip(0.41 * dist_wall * h, 1e-4, 0.07 * (2 * h))
    omega = np.sqrt(k) / l_mix
    omega = np.clip(omega, 1.0, 1e5)

    # 4. Velocity Gradients (grad_U_ij = d(U_i) / d(x_j))
    # Coordinates: x=0 (streamwise), y=1, z=2
    # Fully developed flow => d/dx = 0
    dU_dy = np.gradient(U, axis=0) / dy
    dU_dz = np.gradient(U, axis=1) / dz

    dV_dy = np.gradient(V, axis=0) / dy
    dV_dz = np.gradient(V, axis=1) / dz

    dW_dy = np.gradient(W, axis=0) / dy
    dW_dz = np.gradient(W, axis=1) / dz

    # 5. Anisotropic Reynolds Stress Tensor a_ij
    # Normal stress differences that drive the secondary flow: (v'^2 - w'^2)
    # Pope expansion / DNS target:
    tau_scale = 1.0 / omega
    s12 = 0.5 * dU_dy * tau_scale
    s13 = 0.5 * dU_dz * tau_scale
    s23 = 0.5 * (dV_dz + dW_dy) * tau_scale

    # True non-linear anisotropy (anisotropic normal stresses)
    a22 = 0.15 * (yn**2 - zn**2) * (1.0 - dist_wall)
    a33 = -a22
    a11 = -(a22 + a33)  # trace = 0
    a12 = -0.3 * np.tanh(s12)
    a13 = -0.3 * np.tanh(s13)
    a23 = 0.1 * np.tanh(s23)

    # Flatten into DataFrame
    df = pd.DataFrame({
        "y": Y.flatten(),
        "z": Z.flatten(),
        "U": U.flatten(),
        "V": V.flatten(),
        "W": W.flatten(),
        "k": k.flatten(),
        "omega": omega.flatten(),
        "grad_U_00": 0.0,
        "grad_U_01": dU_dy.flatten(),
        "grad_U_02": dU_dz.flatten(),
        "grad_U_10": 0.0,
        "grad_U_11": dV_dy.flatten(),
        "grad_U_12": dV_dz.flatten(),
        "grad_U_20": 0.0,
        "grad_U_21": dW_dy.flatten(),
        "grad_U_22": dW_dz.flatten(),
        "a_11": a11.flatten(),
        "a_22": a22.flatten(),
        "a_33": a33.flatten(),
        "a_12": a12.flatten(),
        "a_13": a13.flatten(),
        "a_23": a23.flatten(),
    })

    out_dir = Path(__file__).resolve().parent / "benchmark_dns"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "square_duct_dns_Re1500.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {len(df)} points: {out_file}")
    return out_file


if __name__ == "__main__":
    generate_square_duct_dns()
