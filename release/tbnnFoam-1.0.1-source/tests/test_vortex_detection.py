import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validation.vortex_detector import detect_vortex_cores

def generate_synthetic_vortex_field(centers_with_signs, grid_size=65, sigma=0.08):
    y_1d = np.linspace(-1.0, 1.0, grid_size)
    z_1d = np.linspace(-1.0, 1.0, grid_size)
    Z_g, Y_g = np.meshgrid(z_1d, y_1d, indexing='ij')
    field = np.zeros_like(Y_g)
    for yc, zc, sign in centers_with_signs:
        r2 = (Y_g - yc) ** 2 + (Z_g - zc) ** 2
        field += sign * np.exp(-r2 / (2.0 * sigma ** 2))
    return field, y_1d, z_1d

def test_zero_vorticity_field():
    y_1d = np.linspace(-1.0, 1.0, 65)
    z_1d = np.linspace(-1.0, 1.0, 65)
    omega_zero = np.zeros((65, 65))
    cores = detect_vortex_cores(omega_zero, y_1d, z_1d)
    assert len(cores) == 0

def test_known_four_vortex_field():
    known_centers = [
        (-0.5, 0.5, 1.0),
        (0.5, 0.5, -1.0),
        (-0.5, -0.5, -1.0),
        (0.5, -0.5, 1.0)
    ]
    field, y_1d, z_1d = generate_synthetic_vortex_field(known_centers, grid_size=65)
    cores = detect_vortex_cores(field, y_1d, z_1d, min_separation=0.2, min_strength=0.1, min_area=3)
    assert len(cores) == 4
    for yc, zc, sign in known_centers:
        expected_type = 'acw' if sign > 0 else 'cw'
        matched = False
        for c in cores:
            dist = np.sqrt((c['y'] - yc) ** 2 + (c['z'] - zc) ** 2)
            if dist < 0.05 and c['type'] == expected_type:
                matched = True
                break
        assert matched

def test_known_eight_vortex_field():
    known_eight = [
        (-0.5, 0.8, -1.0),
        (0.8, 0.5, -1.0),
        (0.5, -0.8, -1.0),
        (-0.8, -0.5, -1.0),
        (0.5, 0.8, 1.0),
        (0.8, -0.5, 1.0),
        (-0.5, -0.8, 1.0),
        (-0.8, 0.5, 1.0)
    ]
    field, y_1d, z_1d = generate_synthetic_vortex_field(known_eight, grid_size=65)
    cores = detect_vortex_cores(field, y_1d, z_1d, min_separation=0.2, min_strength=0.1, min_area=3)
    assert len(cores) == 8
    cw_cores = [c for c in cores if c['type'] == 'cw']
    acw_cores = [c for c in cores if c['type'] == 'acw']
    assert len(cw_cores) == 4
    assert len(acw_cores) == 4
