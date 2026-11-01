import pytest
import numpy as np
import csv
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

def test_non_square_gradient_ordering():
    # Verify that np.gradient on a 2D array ordered as (Z, Y) requires
    # axis=1 for Y-derivative and axis=0 for Z-derivative.
    Nz, Ny = 30, 20
    z_1d = np.linspace(0.0, 3.0, Nz)
    y_1d = np.linspace(0.0, 2.0, Ny)
    Z, Y = np.meshgrid(z_1d, y_1d, indexing='ij')
    assert Z.shape == (Nz, Ny)
    assert Y.shape == (Nz, Ny)
    
    # f(z, y) = 3*y + 5*z
    F = 3.0 * Y + 5.0 * Z
    dF_dy = np.gradient(F, y_1d, axis=1)
    dF_dz = np.gradient(F, z_1d, axis=0)
    
    assert np.allclose(dF_dy, 3.0, atol=1e-12)
    assert np.allclose(dF_dz, 5.0, atol=1e-12)

def test_non_square_interpolation_ordering():
    # Verify that RegularGridInterpolator with grid points (z_1d, y_1d)
    # requires query points ordered as [z, y].
    Nz, Ny = 15, 25
    z_1d = np.linspace(0.0, 1.5, Nz)
    y_1d = np.linspace(0.0, 2.5, Ny)
    Z, Y = np.meshgrid(z_1d, y_1d, indexing='ij')
    F = 100.0 * Z + 1.0 * Y
    rgi = RegularGridInterpolator((z_1d, y_1d), F, bounds_error=True)
    
    Nz_tar, Ny_tar = 10, 12
    z_tar = np.linspace(0.1, 1.4, Nz_tar)
    y_tar = np.linspace(0.1, 2.4, Ny_tar)
    Z_tar, Y_tar = np.meshgrid(z_tar, y_tar, indexing='ij')
    
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
    fixture_file = repo_root / 'tests' / 'fixtures' / 'synthetic_dns_grid.csv'
    
    target_file = dns_file if dns_file.exists() else fixture_file
    assert target_file.exists(), f"Neither DNS file nor synthetic test fixture found: {target_file}"
    
    rows = []
    with open(target_file, 'r', encoding='utf-8') as f:
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

if __name__ == '__main__':
    pytest.main([__file__])
