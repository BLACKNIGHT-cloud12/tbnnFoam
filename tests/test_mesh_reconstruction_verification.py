
import pytest
from pathlib import Path
import numpy as np

def parse_foam_internal_vec(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    idx = content.find('internalField')
    if idx == -1:
        return np.array([])
    chunk = content[idx:]
    open_paren = chunk.find('(')
    if open_paren == -1:
        return np.array([])
    close_paren = chunk.find('\n)\n;', open_paren)
    if close_paren == -1:
        close_paren = chunk.find('\n);', open_paren)
    if close_paren == -1:
        close_paren = chunk.rfind(')')
    data_str = chunk[open_paren + 1:close_paren].strip()
    vectors = []
    for line in data_str.split('\n'):
        parts = line.strip().strip('()').split()
        if len(parts) == 3:
            try:
                vectors.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                pass
    return np.array(vectors)

def test_coarse_mesh_geometry_and_ordering():
    repo_root = Path(__file__).resolve().parent.parent
    c_file = repo_root / 'verification_squareDuct' / '0' / 'C'
    assert c_file.exists()
    c_raw = parse_foam_internal_vec(c_file)
    assert len(c_raw) == 20000

    Nz, Ny, Nx = 20, 20, 50
    c_3d = c_raw.reshape(Nz, Ny, Nx, 3)
    X = c_3d[:, :, :, 0]
    Y = c_3d[:, :, :, 1]
    Z = c_3d[:, :, :, 2]

    x_1d = X[0, 0, :]
    assert np.all(np.diff(x_1d) > 0)
    for iz in range(Nz):
        for iy in range(Ny):
            assert np.allclose(X[iz, iy, :], x_1d)

    y_1d = Y[0, :, 0]
    assert np.all(np.diff(y_1d) > 0)
    for iz in range(Nz):
        for ix in range(Nx):
            assert np.allclose(Y[iz, :, ix], y_1d)

    z_1d = Z[:, 0, 0]
    assert np.all(np.diff(z_1d) > 0)
    for iy in range(Ny):
        for ix in range(Nx):
            assert np.allclose(Z[:, iy, ix], z_1d)

    h = 0.05
    y_norm = y_1d / h
    z_norm = z_1d / h
    expected_y = np.linspace(-0.95, 0.95, Ny)
    expected_z = np.linspace(-0.95, 0.95, Nz)

    assert np.max(np.abs(y_norm - expected_y)) < 1e-12
    assert np.max(np.abs(z_norm - expected_z)) < 1e-12

def test_fine_mesh_geometry_and_ordering():
    repo_root = Path(__file__).resolve().parent.parent
    c_file = repo_root / 'verification_squareDuct' / 'fine_mesh' / '0' / 'C'
    assert c_file.exists()
    c_raw = parse_foam_internal_vec(c_file)
    assert len(c_raw) == 80000

    Nz, Ny, Nx = 40, 40, 50
    c_3d = c_raw.reshape(Nz, Ny, Nx, 3)
    X = c_3d[:, :, :, 0]
    Y = c_3d[:, :, :, 1]
    Z = c_3d[:, :, :, 2]

    x_1d = X[0, 0, :]
    assert np.all(np.diff(x_1d) > 0)
    for iz in range(Nz):
        for iy in range(Ny):
            assert np.allclose(X[iz, iy, :], x_1d)

    y_1d = Y[0, :, 0]
    assert np.all(np.diff(y_1d) > 0)
    for iz in range(Nz):
        for ix in range(Nx):
            assert np.allclose(Y[iz, :, ix], y_1d)

    z_1d = Z[:, 0, 0]
    assert np.all(np.diff(z_1d) > 0)
    for iy in range(Ny):
        for ix in range(Nx):
            assert np.allclose(Z[:, iy, ix], z_1d)

    h = 0.05
    y_norm = y_1d / h
    z_norm = z_1d / h
    expected_y = np.linspace(-0.975, 0.975, Ny)
    expected_z = np.linspace(-0.975, 0.975, Nz)

    assert np.max(np.abs(y_norm - expected_y)) < 1e-12
    assert np.max(np.abs(z_norm - expected_z)) < 1e-12

if __name__ == '__main__':
    test_coarse_mesh_geometry_and_ordering()
    test_fine_mesh_geometry_and_ordering()
    print('[+] Mesh geometry and reconstruction verification tests passed!')
