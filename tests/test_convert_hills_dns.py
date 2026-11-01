"""
tests/test_convert_hills_dns.py

Unit tests for periodic-hills DNS ingest converter (TASK-02).
"""

from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.convert_hills_dns import (
    HILL_CONTOUR_DEFAULT,
    compute_wall_distance,
    grad_xy,
    default_parser,
    process_hills_data,
)

FROZEN_SCHEMA = [
    "x_over_h",
    "y_over_h",
    "U_over_Ub",
    "V_over_Ub",
    "W_over_Ub",
    "uu",
    "vv",
    "ww",
    "uv",
    "uw",
    "vw",
    "k",
    "omega",
]


def test_wall_distance_flat_segment():
    """Criterion 5: Wall distance: point exactly on a contour segment returns ≈ 0;
    a point 0.1 above a flat segment returns ≈ 0.1."""
    contour_x = np.array([0.0, 1.0, 2.0])
    contour_y = np.array([0.0, 0.0, 0.0])

    # Point exactly on contour segment
    x_on = np.array([0.5])
    y_on = np.array([0.0])
    d_on = compute_wall_distance(x_on, y_on, contour_x, contour_y)
    assert np.isclose(d_on[0], 0.0, atol=1e-12)

    # Point 0.1 above flat segment
    x_above = np.array([0.5])
    y_above = np.array([0.1])
    d_above = compute_wall_distance(x_above, y_above, contour_x, contour_y)
    assert np.isclose(d_above[0], 0.1, atol=1e-12)


def test_wall_distance_concave_and_shape_preservation():
    """Requirement 1: compute_wall_distance must handle concave contours
    (point-to-SEGMENT distance, not point-to-vertex) and return 0 at wall-adjacent points.
    Also verify 2D shape preservation."""
    # V-shaped concave contour: (0, 1) -> (1, 0) -> (2, 1)
    contour_x = [0.0, 1.0, 2.0]
    contour_y = [1.0, 0.0, 1.0]

    # Query point at (1.0, 0.5):
    # Closest point on segment (0,1)-(1,0) is (0.75, 0.25), distance = sqrt(0.25^2 + 0.25^2) = 0.25*sqrt(2) ≈ 0.35355.
    # Distance to vertex (1, 0) would be 0.5. Point-to-segment distance is smaller!
    query_x = np.array([[1.0, 0.5], [1.5, 1.0]])
    query_y = np.array([[0.5, 0.5], [0.5, 0.0]])

    d = compute_wall_distance(query_x, query_y, contour_x, contour_y)
    assert d.shape == (2, 2)
    expected_dist_concave = 0.25 * np.sqrt(2.0)
    assert np.isclose(d[0, 0], expected_dist_concave, atol=1e-12)

    # Point exactly on vertex (1.0, 0.0)
    assert np.isclose(d[1, 1], 0.0, atol=1e-12)


def test_grad_xy_uniform_subgrid():
    """Criterion 6: grad_xy on a uniform subgrid: d/dx of sin(k*x) matches cos(k*x)*k
    to 1e-6 in the interior."""
    k_w = 1.0
    nx, ny = 1000, 20
    x_1d = np.linspace(0.0, 1.0, nx)
    y_1d = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x_1d, y_1d)

    field = np.sin(k_w * X)
    true_dfdx = k_w * np.cos(k_w * X)

    dfdx, dfdy = grad_xy(field, X, Y)

    # Check interior error along x (axis 1)
    interior_err = np.max(np.abs(dfdx[:, 1:-1] - true_dfdx[:, 1:-1]))
    assert interior_err < 1e-6, f"Interior error {interior_err} exceeded 1e-6"


def test_grad_xy_stretched_nonuniform_second_order():
    """Requirement 2: grad_xy non-uniform grid support with one-sided second-order
    stencils at edges. A 2nd-order scheme is exact to machine precision for quadratic polynomials."""
    ny, nx = 25, 30
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    # Stretched coordinates
    x_1d = xi**1.6 * 4.0
    y_1d = eta**1.4 * 2.5
    X, Y = np.meshgrid(x_1d, y_1d)

    # 2D quadratic field
    a, b, c, d, e = 2.5, -1.3, 3.2, 0.8, 1.1
    field = a * X**2 + b * X + c * Y**2 + d * Y + e
    exact_dfdx = 2.0 * a * X + b
    exact_dfdy = 2.0 * c * Y + d

    dfdx, dfdy = grad_xy(field, X, Y)

    # Should be exact across the entire domain including boundaries
    assert np.max(np.abs(dfdx - exact_dfdx)) < 1e-11
    assert np.max(np.abs(dfdy - exact_dfdy)) < 1e-11


@pytest.fixture
def synthetic_hills_case(tmp_path):
    """Generate synthetic periodic-hills raw DNS file:
    Criterion 1: 20x30 grid, smooth analytic U = 0.5 + 0.1*sin(pi*y/H), flat-ish contour, known stresses.
    """
    ny = 20
    nx = 30
    H = 3.0
    h_ref = 1.0
    u_ref = 1.0

    x_1d = np.linspace(0.0, 9.0, nx)
    # Flat-ish lower wall contour
    contour_x = np.linspace(0.0, 9.0, 15)
    contour_y = 0.08 * np.sin(np.pi * contour_x / 9.0)

    # Interpolate contour to x_1d
    y_wall_x = np.interp(x_1d, contour_x, contour_y)

    # Construct 2D grid (ny, nx)
    X = np.empty((ny, nx), dtype=float)
    Y = np.empty((ny, nx), dtype=float)
    for j in range(nx):
        X[:, j] = x_1d[j]
        # Stretched spacing in y
        eta = np.linspace(0.0, 1.0, ny) ** 1.2
        Y[:, j] = y_wall_x[j] + eta * (H - y_wall_x[j])

    # Analytic fields
    U = 0.5 + 0.1 * np.sin(np.pi * Y / H)
    V = 0.02 * np.cos(np.pi * X / 9.0)
    W = np.zeros_like(U)
    uu = 0.05 * (1.0 + 0.1 * np.cos(np.pi * Y / H))
    vv = 0.02 * (1.0 + 0.1 * np.cos(np.pi * Y / H))
    ww = 0.03 * (1.0 + 0.1 * np.cos(np.pi * Y / H))
    uv = -0.008 * np.ones_like(U)
    uw = np.zeros_like(U)
    vw = np.zeros_like(U)

    # Write synthetic raw file
    raw_path = tmp_path / "synthetic_hills_dns.raw"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(f"# n_y={ny}, n_x={nx}\n")
        for i in range(ny):
            for j in range(nx):
                f.write(
                    f"{X[i, j]:.8e} {Y[i, j]:.8e} {U[i, j]:.8e} {V[i, j]:.8e} {W[i, j]:.8e} "
                    f"{uu[i, j]:.8e} {vv[i, j]:.8e} {ww[i, j]:.8e} {uv[i, j]:.8e} {uw[i, j]:.8e} {vw[i, j]:.8e}\n"
                )

    # Save contour to file for testing
    contour_path = tmp_path / "contour.txt"
    np.savetxt(contour_path, np.column_stack([contour_x, contour_y]))

    return {
        "raw_path": raw_path,
        "contour_path": contour_path,
        "contour_pts": (contour_x, contour_y),
        "ny": ny,
        "nx": nx,
        "h_ref": h_ref,
        "u_ref": u_ref,
    }


def test_synthetic_hills_conversion_and_schema(synthetic_hills_case, tmp_path):
    """Criteria 1, 2, 3, 4:
    - Build synthetic raw file: 20x30, smooth analytic U, flat-ish contour, known stresses.
    - Convert -> DataFrame.
    - Assert column names and order exactly match the frozen schema.
    - Assert k == 0.5*(uu+vv+ww) to 1e-12 on every row.
    - Assert omega > 0 everywhere.
    """
    raw_path = synthetic_hills_case["raw_path"]
    contour_pts = synthetic_hills_case["contour_pts"]
    h_ref = synthetic_hills_case["h_ref"]
    u_ref = synthetic_hills_case["u_ref"]
    out_csv = tmp_path / "output_hills.csv"

    # Convert using process_hills_data
    df = process_hills_data(
        raw_parser=default_parser,
        raw_path=raw_path,
        out_csv_path=out_csv,
        h_ref=h_ref,
        u_ref=u_ref,
        contour=contour_pts,
    )

    # Criterion 2: Assert column names and order exactly match frozen schema
    assert list(df.columns) == FROZEN_SCHEMA, (
        f"Columns mismatch: expected {FROZEN_SCHEMA}, got {list(df.columns)}"
    )

    # Verify number of rows is ny * nx = 600
    expected_rows = synthetic_hills_case["ny"] * synthetic_hills_case["nx"]
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    # Criterion 3: Assert k == 0.5*(uu+vv+ww) to 1e-12 on every row
    k_expected = 0.5 * (df["uu"] + df["vv"] + df["ww"])
    k_diff = np.abs(df["k"] - k_expected)
    assert (k_diff <= 1e-12).all(), f"k identity failed, max diff = {k_diff.max()}"

    # Criterion 4: Assert omega > 0 everywhere
    assert (df["omega"] > 0).all(), "omega must be strictly positive everywhere"
    assert not df["omega"].isna().any(), "omega must not contain NaNs"
    assert not np.isinf(df["omega"]).any(), "omega must not contain Infs"

    # Verify output file was written and can be read back cleanly
    assert out_csv.exists()
    df_read = pd.read_csv(out_csv)
    assert list(df_read.columns) == FROZEN_SCHEMA
    assert np.allclose(df_read["k"], df["k"], atol=1e-12)
    assert np.allclose(df_read["omega"], df["omega"], atol=1e-12)


def test_cli_execution(synthetic_hills_case, tmp_path):
    """Test convert_hills_dns.py via command-line interface."""
    raw_path = synthetic_hills_case["raw_path"]
    contour_path = synthetic_hills_case["contour_path"]
    out_csv = tmp_path / "cli_output.csv"

    cmd = [
        sys.executable,
        "scripts/convert_hills_dns.py",
        "--raw",
        str(raw_path),
        "--out",
        str(out_csv),
        "--contour",
        str(contour_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert out_csv.exists(), "CLI output CSV was not created"

    df = pd.read_csv(out_csv)
    assert list(df.columns) == FROZEN_SCHEMA
    assert len(df) == synthetic_hills_case["ny"] * synthetic_hills_case["nx"]
    assert (df["omega"] > 0).all()
