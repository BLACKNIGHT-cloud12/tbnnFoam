# TASK-02 — Periodic-hills DNS ingest converter (schema-frozen, parser pluggable)

Fully-specified module. Do not redesign. Do not touch files outside the list.

## Context

tbnnFoam v2 trains on the open periodic-hills DNS (Krank, Kronbichler & Wall 2018, TUM,
DOI 10.14459/2018mp1415670). The dataset download and its exact raw-file format are handled
separately; therefore the **parser is pluggable** but the **output schema is frozen now** —
the training pipeline will be written against this CSV schema, so it must not drift.
Precedent: the duct converter `scripts/convert_real_dns.py` produces
`square_duct_dns_Reb4410_Retau150.csv` for `tbnn_train.py`; you are building the hills
equivalent.

## Files to create (ONLY these two)

1. `scripts/convert_hills_dns.py`
2. `tests/test_convert_hills_dns.py`

## Interfaces (implement exactly)

```python
def compute_wall_distance(x, y, contour_x, contour_y):
    """Distance from each (x, y) point to the lower wall contour defined by
    (contour_x, contour_y) piecewise-linear points. Distance is the minimum over
    contour segments of the point-to-segment distance. x, y are 2D arrays
    (n_y, n_x) or flattened 1D; return array of matching shape."""

def grad_xy(field, x, y):
    """Second-order partial derivatives (d/dx, d/dy) on a structured grid with
    NON-UNIFORM spacing. field, x, y are 2D arrays of shape (n_y, n_x).
    Returns (dfdx, dfdy), same shape. Use one-sided second-order stencils at edges."""

def process_hills_data(raw_parser, raw_path, out_csv_path, h_ref, u_ref):
    """Convert raw DNS plane data to the frozen training CSV.
    raw_parser: callable(raw_path) -> dict of numpy arrays with keys
        'x', 'y', 'U', 'V', 'W', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw'
        (2D arrays, shape (n_y, n_x); stresses are plane-averaged Reynolds stresses).
    h_ref: hill height; u_ref: bulk velocity.
    Writes out_csv_path and returns the pandas DataFrame."""

def default_parser(raw_path):
    """Reference parser for the documented raw format (below). Other formats are
    adapted later by adding new parser callables — the schema does not change."""
```

**Reference raw format for `default_parser`** (white-space separated, header line starting
with `#` giving n_y and n_x, then n_y*n_x rows: `x y U V W uu vv ww uv uw vw`):
document it in the module docstring verbatim.

## Frozen output CSV schema (exact column names, in order)

```
x_over_h, y_over_h, U_over_Ub, V_over_Ub, W_over_Ub, uu, vv, ww, uv, uw, vw, k, omega
```

- `x_over_h = x / h_ref`, `y_over_h = y / h_ref`
- velocities divided by `u_ref`; Reynolds stresses divided by `u_ref**2`
- `k = 0.5*(uu + vv + ww)` (computed post-normalization)
- `omega` = SST surrogate: `omega = sqrt(k) / (C_mu**0.25 * l)` with `C_mu = 0.09`,
  `kappa = 0.41`, `l = min(kappa * d_w, 0.07 * D_h)`, `D_h = channel height at that x minus
  y_wall(x)` computed from the contour, `d_w` from `compute_wall_distance` against the
  supplied contour file. The contour is passed via a module-level constant
  `HILL_CONTOUR_DEFAULT` (a small built-in piecewise-linear placeholder (x, y) list clearly
  marked for replacement when the exact TUM contour lands) plus an optional
  `contour_file` argument on the CLI.

## CLI

```
python scripts/convert_hills_dns.py --raw RAW_PATH --out OUT_CSV [--contour CONTOUR_FILE]
```

## Logic requirements

1. `compute_wall_distance` must handle concave contours (point-to-SEGMENT distance, not
   point-to-vertex) and return 0 at wall-adjacent points.
2. `grad_xy` non-uniform grid support is mandatory (hills grids are stretched).
3. k identity must hold exactly in the CSV (recompute from normalized stresses).
4. omega strictly positive; clip k at a floor of 1e-12 before sqrt.
5. Vectorized numpy; no loops over grid points where vectorization is natural.

## Constraints

- Dependencies: numpy, pandas (repo pins). No network. LF line endings, no emoji.
- Do NOT modify any existing file (including `convert_real_dns.py` and `tbnn_train.py`).

## Acceptance criteria

Synthetic tests only (no real DNS data):

1. Build a synthetic raw file in the documented format: 20×30 grid, smooth analytic
   U = 0.5 + 0.1*sin(pi*y/H), flat-ish contour, known stresses. Convert → DataFrame.
2. Assert column names and order exactly match the frozen schema.
3. Assert `k == 0.5*(uu+vv+ww)` to 1e-12 on every row.
4. Assert omega > 0 everywhere.
5. Wall distance: point exactly on a contour segment returns ≈ 0; a point 0.1 above a flat
   segment returns ≈ 0.1.
6. `grad_xy`: on a uniform subgrid, d/dx of sin(k*x) matches cos(k*x)*k to 1e-6 in the
   interior.

Run and record:

```
pytest tests/test_convert_hills_dns.py -v    # all PASS
pytest tests/ -q                             # existing suite still green
```

## When done

Write `parallel-tasks/TASK-02-RESULT.md`: files created, pytest summary lines, deviations (none expected).
