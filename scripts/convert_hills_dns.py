"""
scripts/convert_hills_dns.py

Periodic-hills DNS ingest converter (schema-frozen, parser pluggable).

Reference raw format for default_parser (white-space separated, header line starting
with # giving n_y and n_x, then n_y*n_x rows: x y U V W uu vv ww uv uw vw):
document it in the module docstring verbatim.
"""

from pathlib import Path
import re
import sys
import zipfile
import numpy as np
import pandas as pd


# Default placeholder piecewise-linear contour (x, y) for the lower hill wall.
# Replace with exact TUM periodic hills contour (Krank et al. 2018) when available.
HILL_CONTOUR_DEFAULT = [
    (0.0, 1.0),
    (0.5, 0.9),
    (1.0, 0.7),
    (1.5, 0.35),
    (2.0, 0.1),
    (2.5, 0.0),
    (6.5, 0.0),
    (7.0, 0.1),
    (7.5, 0.35),
    (8.0, 0.7),
    (8.5, 0.9),
    (9.0, 1.0),
]


def compute_wall_distance(x, y, contour_x, contour_y):
    """Distance from each (x, y) point to the lower wall contour defined by
    (contour_x, contour_y) piecewise-linear points. Distance is the minimum over
    contour segments of the point-to-segment distance. x, y are 2D arrays
    (n_y, n_x) or flattened 1D; return array of matching shape."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    orig_shape = x_arr.shape

    cx = np.asarray(contour_x, dtype=float).ravel()
    cy = np.asarray(contour_y, dtype=float).ravel()
    if len(cx) != len(cy):
        raise ValueError("contour_x and contour_y must have the same length")
    if len(cx) < 2:
        raise ValueError("Contour must contain at least 2 points to form a segment")

    # Segment start and end points
    px0 = cx[:-1]
    py0 = cy[:-1]
    px1 = cx[1:]
    py1 = cy[1:]

    dx_seg = px1 - px0
    dy_seg = py1 - py0
    seg_len_sq = dx_seg**2 + dy_seg**2

    # Avoid division by zero for degenerate segments
    safe_seg_len_sq = np.where(seg_len_sq > 0, seg_len_sq, 1.0)

    # Flatten query points: shape (P, 1)
    x_flat = x_arr.ravel()[:, np.newaxis]
    y_flat = y_arr.ravel()[:, np.newaxis]

    # Segments shape: (1, M)
    px0 = px0[np.newaxis, :]
    py0 = py0[np.newaxis, :]
    dx_seg = dx_seg[np.newaxis, :]
    dy_seg = dy_seg[np.newaxis, :]
    safe_seg_len_sq = safe_seg_len_sq[np.newaxis, :]
    seg_len_sq = seg_len_sq[np.newaxis, :]

    # Vector from segment start to query point
    vx = x_flat - px0
    vy = y_flat - py0

    # Projection parameter t clamped to [0, 1]
    t = (vx * dx_seg + vy * dy_seg) / safe_seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    t = np.where(seg_len_sq > 0, t, 0.0)

    # Closest point on segment
    proj_x = px0 + t * dx_seg
    proj_y = py0 + t * dy_seg

    # Distance squared to each segment
    dist_sq = (x_flat - proj_x)**2 + (y_flat - proj_y)**2
    min_dist_sq = np.min(dist_sq, axis=1)
    min_dist = np.sqrt(np.maximum(min_dist_sq, 0.0))

    return min_dist.reshape(orig_shape)


def grad_xy(field, x, y):
    """Second-order partial derivatives (d/dx, d/dy) on a structured grid with
    NON-UNIFORM spacing. field, x, y are 2D arrays of shape (n_y, n_x).
    Returns (dfdx, dfdy), same shape. Use one-sided second-order stencils at edges."""
    f = np.asarray(field, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if f.ndim != 2 or x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("field, x, and y must all be 2D arrays")
    if f.shape != x_arr.shape or f.shape != y_arr.shape:
        raise ValueError(
            f"field, x, and y must have the same shape, got {f.shape}, {x_arr.shape}, {y_arr.shape}"
        )

    n_y, n_x = f.shape
    if n_y < 3 or n_x < 3:
        raise ValueError("Grid must have at least 3 points along each axis for 2nd order stencil")

    def _diff_axis(f_arr, coord, axis):
        f_m = np.moveaxis(f_arr, axis, -1)
        c_m = np.moveaxis(coord, axis, -1)
        df = np.empty_like(f_m, dtype=float)

        # Interior: second-order central difference for non-uniform grid
        hm = c_m[..., 1:-1] - c_m[..., :-2]
        hp = c_m[..., 2:] - c_m[..., 1:-1]
        denom = hm * hp * (hm + hp)
        df[..., 1:-1] = (
            hm**2 * (f_m[..., 2:] - f_m[..., 1:-1]) - hp**2 * (f_m[..., :-2] - f_m[..., 1:-1])
        ) / denom

        # Left edge: second-order forward difference for non-uniform grid
        d1 = c_m[..., 1] - c_m[..., 0]
        d2 = c_m[..., 2] - c_m[..., 0]
        df[..., 0] = (
            d2**2 * (f_m[..., 1] - f_m[..., 0]) - d1**2 * (f_m[..., 2] - f_m[..., 0])
        ) / (d1 * d2 * (d2 - d1))

        # Right edge: second-order backward difference for non-uniform grid
        d1 = c_m[..., -1] - c_m[..., -2]
        d2 = c_m[..., -1] - c_m[..., -3]
        df[..., -1] = (
            -d2**2 * (f_m[..., -2] - f_m[..., -1]) + d1**2 * (f_m[..., -3] - f_m[..., -1])
        ) / (d1 * d2 * (d2 - d1))

        return np.moveaxis(df, -1, axis)

    dfdx = _diff_axis(f, x_arr, axis=1)
    dfdy = _diff_axis(f, y_arr, axis=0)

    return dfdx, dfdy


def default_parser(raw_path):
    """Reference parser for the documented raw format (below). Other formats are
    adapted later by adding new parser callables — the schema does not change.

    Reference raw format for default_parser (white-space separated, header line starting
    with # giving n_y and n_x, then n_y*n_x rows: x y U V W uu vv ww uv uw vw)
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw DNS file not found: {raw_path}")

    header = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s.startswith("#"):
                header = line_s
                break

    if not header:
        raise ValueError(f"No header line starting with '#' found in {raw_path}")

    # Extract n_y and n_x
    m_ny = re.search(r"n_y\s*[:=]?\s*(\d+)", header, re.IGNORECASE)
    m_nx = re.search(r"n_x\s*[:=]?\s*(\d+)", header, re.IGNORECASE)
    if m_ny and m_nx:
        n_y = int(m_ny.group(1))
        n_x = int(m_nx.group(1))
    else:
        ints = [int(s) for s in re.findall(r"\b\d+\b", header)]
        if len(ints) >= 2:
            n_y, n_x = ints[0], ints[1]
        else:
            raise ValueError(f"Could not parse n_y and n_x from header: {header}")

    data = np.loadtxt(path, comments="#")
    expected_rows = n_y * n_x
    if data.ndim == 1 and expected_rows == 1:
        data = data.reshape(1, -1)
    if data.shape != (expected_rows, 11):
        raise ValueError(
            f"Expected data shape ({expected_rows}, 11) for n_y={n_y}, n_x={n_x}, got {data.shape}"
        )

    col_names = ["x", "y", "U", "V", "W", "uu", "vv", "ww", "uv", "uw", "vw"]
    parsed = {}
    for idx, name in enumerate(col_names):
        parsed[name] = data[:, idx].reshape(n_y, n_x)

    return parsed


def process_hills_data(raw_parser, raw_path, out_csv_path, h_ref=1.0, u_ref=1.0, contour=None):
    """Convert raw DNS plane data to the frozen training CSV.
    raw_parser: callable(raw_path) -> dict of numpy arrays with keys
        'x', 'y', 'U', 'V', 'W', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw'
        (2D arrays, shape (n_y, n_x); stresses are plane-averaged Reynolds stresses).
    h_ref: hill height; u_ref: bulk velocity.
    Writes out_csv_path and returns the pandas DataFrame."""
    raw = raw_parser(raw_path)
    required_keys = ["x", "y", "U", "V", "W", "uu", "vv", "ww", "uv", "uw", "vw"]
    for k_key in required_keys:
        if k_key not in raw:
            raise KeyError(f"raw_parser output missing required key: '{k_key}'")

    x = np.asarray(raw["x"], dtype=float)
    y = np.asarray(raw["y"], dtype=float)
    U = np.asarray(raw["U"], dtype=float)
    V = np.asarray(raw["V"], dtype=float)
    W = np.asarray(raw["W"], dtype=float)
    uu = np.asarray(raw["uu"], dtype=float)
    vv = np.asarray(raw["vv"], dtype=float)
    ww = np.asarray(raw["ww"], dtype=float)
    uv = np.asarray(raw["uv"], dtype=float)
    uw = np.asarray(raw["uw"], dtype=float)
    vw = np.asarray(raw["vw"], dtype=float)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D arrays from raw_parser, got {x.ndim}D for 'x'")
    grid_shape = x.shape
    for name, arr in [
        ("y", y),
        ("U", U),
        ("V", V),
        ("W", W),
        ("uu", uu),
        ("vv", vv),
        ("ww", ww),
        ("uv", uv),
        ("uw", uw),
        ("vw", vw),
    ]:
        if arr.shape != grid_shape:
            raise ValueError(f"Shape mismatch: '{name}' has shape {arr.shape}, expected {grid_shape}")

    # Normalization
    h_ref_f = float(h_ref)
    u_ref_f = float(u_ref)
    x_over_h = x / h_ref_f
    y_over_h = y / h_ref_f
    U_over_Ub = U / u_ref_f
    V_over_Ub = V / u_ref_f
    W_over_Ub = W / u_ref_f

    u_ref_sq = u_ref_f**2
    uu_norm = uu / u_ref_sq
    vv_norm = vv / u_ref_sq
    ww_norm = ww / u_ref_sq
    uv_norm = uv / u_ref_sq
    uw_norm = uw / u_ref_sq
    vw_norm = vw / u_ref_sq

    # Recompute k post-normalization
    k = 0.5 * (uu_norm + vv_norm + ww_norm)

    # Contour handling
    if contour is None:
        contour_pts = np.asarray(HILL_CONTOUR_DEFAULT, dtype=float)
    elif isinstance(contour, (str, Path)):
        contour_pts = np.loadtxt(contour)
    elif isinstance(contour, tuple) and len(contour) == 2:
        contour_pts = np.column_stack(
            [np.asarray(contour[0], dtype=float), np.asarray(contour[1], dtype=float)]
        )
    else:
        contour_pts = np.asarray(contour, dtype=float)

    contour_x = contour_pts[:, 0]
    contour_y = contour_pts[:, 1]

    # If contour is given in raw physical coordinates (not normalized), scale to match x_over_h
    if np.max(contour_x) > 1.5 * np.max(x_over_h) and h_ref_f != 1.0:
        cx_norm = contour_x / h_ref_f
        cy_norm = contour_y / h_ref_f
    else:
        cx_norm = contour_x
        cy_norm = contour_y

    d_w = compute_wall_distance(x_over_h, y_over_h, cx_norm, cy_norm)

    # Local channel height at that x: channel height at that x minus y_wall(x) computed from contour
    y_top = np.max(y_over_h, axis=0, keepdims=True)
    y_wall = np.interp(x_over_h, cx_norm, cy_norm)
    D_h = y_top - y_wall
    D_h = np.maximum(D_h, 1e-12)

    # SST surrogate omega calculation
    C_mu = 0.09
    kappa = 0.41
    l_scale = np.minimum(kappa * d_w, 0.07 * D_h)
    l_scale = np.maximum(l_scale, 1e-12)

    k_clipped = np.maximum(k, 1e-12)
    omega = np.sqrt(k_clipped) / ((C_mu**0.25) * l_scale)

    df = pd.DataFrame(
        {
            "x_over_h": x_over_h.ravel(),
            "y_over_h": y_over_h.ravel(),
            "U_over_Ub": U_over_Ub.ravel(),
            "V_over_Ub": V_over_Ub.ravel(),
            "W_over_Ub": W_over_Ub.ravel(),
            "uu": uu_norm.ravel(),
            "vv": vv_norm.ravel(),
            "ww": ww_norm.ravel(),
            "uv": uv_norm.ravel(),
            "uw": uw_norm.ravel(),
            "vw": vw_norm.ravel(),
            "k": k.ravel(),
            "omega": omega.ravel(),
        }
    )

    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, lineterminator="\n")
    return df


def voet_case_parser(raw_path):
    """Parser for the Voet & Laizet / Xiao et al. (2020) periodic-hills DNS archives.

    raw_path format: "<archive.zip>::<member.dat>" — e.g.
        "voet_raw/DNS Data/Case 2/Archive.zip::mean_files.dat"
    (subdirectory members work, e.g. "::BIG_XYZ/mean_files.dat").

    Column mapping (verified on the reference case by the turbulent-kinetic-energy
    identity k = 0.5*(uu+vv+ww), all coordinates in hill-height units, velocities
    normalized by the bulk velocity):
      mean_files.dat: x y U V W (column 6 unused here)
      rms_files1.dat: x y uu vv ww uv
      rms_files2.dat: x y uw vw (column 5 unused here)

    Returns the standard 11-key dict of (n_y, n_x) arrays consumed by
    process_hills_data. Rows are x-fastest within each y, matching the
    reshape(n_y, n_x) convention of default_parser.
    """
    if "::" not in str(raw_path):
        raise ValueError(
            "voet_case_parser expects '<archive.zip>::<member.dat>', got: %s" % raw_path
        )
    archive_str, member = str(raw_path).split("::", 1)
    archive = Path(archive_str)
    if not archive.exists():
        raise FileNotFoundError(f"Voet/Laizet archive not found: {archive}")

    with zipfile.ZipFile(archive) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"Member '{member}' not in archive {archive}")

        def load(shape_cols, name):
            with zf.open(name) as f:
                return np.loadtxt(f)

        mean = load(6, member)
        folder, stem = member.rsplit("/", 1) if "/" in member else ("", member)
        tag = stem[len("mean_files"):-len(".dat")]  # "" or e.g. "_DNS2"
        prefix = f"{folder}/" if folder else ""
        r1_name = f"{prefix}rms_files1{tag}.dat"
        r2_name = f"{prefix}rms_files2{tag}.dat"
        rms1 = load(6, r1_name)
        rms2 = load(5, r2_name)

    if mean.shape[0] != rms1.shape[0] or mean.shape[0] != rms2.shape[0]:
        raise ValueError(
            f"Row-count mismatch: mean={mean.shape[0]}, rms1={rms1.shape[0]}, rms2={rms2.shape[0]}"
        )

    n_x = len(np.unique(mean[:, 0]))
    if mean.shape[0] % n_x != 0:
        raise ValueError(f"Row count {mean.shape[0]} not divisible by n_x={n_x}")
    n_y = mean.shape[0] // n_x

    def plane(col):
        return col.reshape(n_y, n_x)

    return {
        "x": plane(mean[:, 0]),
        "y": plane(mean[:, 1]),
        "U": plane(mean[:, 2]),
        "V": plane(mean[:, 3]),
        "W": plane(mean[:, 4]),
        "uu": plane(rms1[:, 2]),
        "vv": plane(rms1[:, 3]),
        "ww": plane(rms1[:, 4]),
        "uv": plane(rms1[:, 5]),
        "uw": plane(rms2[:, 2]),
        "vw": plane(rms2[:, 3]),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert periodic-hills DNS raw data to standardized CSV"
    )
    parser.add_argument("--raw", type=Path, required=True, help="Path to raw DNS data file")
    parser.add_argument("--out", type=Path, required=True, help="Path to output CSV file")
    parser.add_argument(
        "--contour", type=Path, default=None, help="Path to hill contour file (optional)"
    )
    parser.add_argument(
        "--format",
        choices=["default", "voet"],
        default="default",
        help="Raw format: 'default' (documented plain format) or 'voet' "
        "(Voet & Laizet / Xiao et al. 2020 archive, raw_path = '<zip>::<member>')",
    )
    parser.add_argument(
        "--href",
        "--h_ref",
        dest="href",
        type=float,
        default=1.0,
        help="Hill reference height (default: 1.0)",
    )
    parser.add_argument(
        "--uref",
        "--u_ref",
        dest="uref",
        type=float,
        default=1.0,
        help="Bulk reference velocity (default: 1.0)",
    )
    args = parser.parse_args()

    process_hills_data(
        raw_parser=voet_case_parser if args.format == "voet" else default_parser,
        raw_path=args.raw,
        out_csv_path=args.out,
        h_ref=args.href,
        u_ref=args.uref,
        contour=args.contour,
    )
