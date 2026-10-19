import csv
from pathlib import Path
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

def test_non_square_gradient_ordering():
    Nz, Ny = 15, 23
    z = np.linspace(-1.0, 1.0, Nz)
    y = np.linspace(-2.0, 2.0, Ny)
    Z, Y = np.meshgrid(z, y, indexing='ij')
    assert Z.shape == (Nz, Ny)
    assert Y.shape == (Nz, Ny)
    f = 100.0 * Z + 1.0 * Y
    dz = z[1] - z[0]
    dy = y[1] - y[0]
    df_dz = np.gradient(f, axis=0) / dz
    df_dy = np.gradient(f, axis=1) / dy
    assert np.allclose(df_dz, 100.0)
    assert np.allclose(df_dy, 1.0)
    swapped_df_dz = np.gradient(f, axis=1) / dz
    swapped_df_dy = np.gradient(f, axis=0) / dy
    assert not np.allclose(swapped_df_dz, 100.0)
    assert not np.allclose(swapped_df_dy, 1.0)
    assert np.max(np.abs(swapped_df_dz - 100.0)) > 90.0

def test_non_square_interpolation_ordering():
    Nz, Ny = 15, 23
    z = np.linspace(-1.0, 1.0, Nz)
    y = np.linspace(-2.0, 2.0, Ny)
    Z, Y = np.meshgrid(z, y, indexing='ij')
    f = 100.0 * Z + 1.0 * Y
    rgi = RegularGridInterpolator((z, y), f, bounds_error=True)
    Nz_tar, Ny_tar = 31, 17
    z_tar = np.linspace(-0.8, 0.8, Nz_tar)
    y_tar = np.linspace(-1.5, 1.5, Ny_tar)
    Z_tar, Y_tar = np.meshgrid(z_tar, y_tar, indexing='ij')
    assert Z_tar.shape == (Nz_tar, Ny_tar)
    assert Y_tar.shape == (Nz_tar, Ny_tar)
    eval_pts_correct = np.column_stack([Z_tar.flatten(), Y_tar.flatten()])
    f_interp = rgi(eval_pts_correct).reshape(Nz_tar, Ny_tar)
    f_exact = 100.0 * Z_tar + 1.0 * Y_tar
    assert np.allclose(f_interp, f_exact, atol=1e-10)
    eval_pts_swapped = np.column_stack([Y_tar.flatten(), Z_tar.flatten()])
    with pytest.raises(ValueError):
        rgi(eval_pts_swapped)

def test_benchmark_dns_grid_convention():
    repo_root = Path(__file__).resolve().parent.parent
    dns_file = repo_root / 'data' / 'benchmark_dns' / 'square_duct_dns_Reb4410_Retau150.csv'
    assert dns_file.exists()
    rows = []
    with open(dns_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    Nz = int(round(np.sqrt(len(rows))))
    Ny = Nz
    assert len(rows) == Nz * Ny
    dns_z = np.array([float(r['z']) for r in rows]).reshape(Nz, Ny)
    dns_y = np.array([float(r['y']) for r in rows]).reshape(Nz, Ny)
    z_1d = np.unique(dns_z)
    y_1d = np.unique(dns_y)
    assert len(z_1d) == Nz
    assert len(y_1d) == Ny
    for i in range(Nz):
        assert np.allclose(dns_z[i, :], dns_z[i, 0])
    for j in range(Ny):
        assert np.allclose(dns_y[:, j], dns_y[0, j])
