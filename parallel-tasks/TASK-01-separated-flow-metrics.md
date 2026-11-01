# TASK-01 — Separated-flow validation metrics module

You are implementing a fully-specified module for the tbnnFoam repository. Everything you need
is in this brief — do not redesign interfaces, do not touch files outside the list.

## Context

tbnnFoam is an OpenFOAM-based tensor-basis neural-network RANS solver. v1 validated secondary
flow in square ducts using an algorithmic vortex detector (`scripts/validation/vortex_detector.py`).
v2 targets **separated flow over periodic hills**, which needs objective algorithmic metrics:
separation point, reattachment point/length, and recirculation-bubble strength, derived from
wall shear stress and velocity fields. Your module is the post-processing backbone for that
validation. It must run without OpenFOAM or DNS data (pure numpy/scipy), so it passes CI.

## Files to create (ONLY these two)

1. `scripts/validation/separated_flow_metrics.py`
2. `tests/test_separated_flow_metrics.py`

## Interfaces (implement exactly)

```python
def wall_shear_events(tau_w, x, min_persistence=2):
    """Classify sign changes of wall shear stress along a streamwise coordinate.

    Parameters
    ----------
    tau_w : 1D array, wall shear stress (Pa), ordered by increasing x.
            tau_w > 0 = forward flow; tau_w < 0 = reversed flow (separated).
    x : 1D array, streamwise coordinates (monotone, may be non-uniform).
    min_persistence : int, a sign change counts only if the new sign persists
        for at least this many consecutive samples (noise rejection).

    Returns
    -------
    list of (kind, x_event) tuples in order, kind in {"separation", "reattachment"}.
    "separation"  = crossing + -> - ; "reattachment" = crossing - -> + .
    """

def separation_bubbles(tau_w, x, min_persistence=2):
    """Return list of (x_sep, x_re) tuples, one per separation bubble."""

def reattachment_length(bubble):
    """x_re - x_sep for one (x_sep, x_re) tuple."""

def bubble_strength(y, u, x, x_sep, x_re):
    """Peak reverse-flow speed inside the bubble: max |min(u, 0)| over all y and
    x in [x_sep, x_re] (u is 2D with shape (n_y, n_x), increasing x along axis 1).
    Returns 0.0 if no negative u exists in the window."""

def friction_velocity_from_gradient(y_wall_layer, u_wall_layer, nu):
    """u_tau = sqrt(nu * max(du/dy|_wall, 0)) using second-order one-sided
    finite differences from the first three wall-normal samples.
    y_wall_layer: first 3 wall-normal coords; u_wall_layer: velocity at those coords."""

def detect_from_wall_data(x, tau_w, y=None, u=None, nu=None, min_persistence=2):
    """Convenience report. Returns dict:
    {"bubbles": [...], "reattachment_lengths": [...], "bubble_strengths": [...],
     "u_tau": float or None (computed when y/u/nu given, at x[0])}"""
```

## Logic requirements

1. Sign-change detection must use persistence filtering exactly as documented: a single
   sample of flipped sign inside an otherwise steady-sign region produces NO event.
2. Handle non-uniform x spacing (no assumption of constant dx anywhere).
3. An oscillating tau_w at the very first or last sample must not create spurious events.
4. `friction_velocity_from_gradient`: use the standard second-order one-sided formula
   (-3u0 + 4u1 - u2) / (2 dy) with dy from the actual three points (support non-uniform dy).
5. No global state; pure functions; numpy only (no torch, no OpenFOAM imports).

## Constraints

- Dependencies: numpy (repo pins in `requirements.txt`). No network access.
- LF line endings; no emoji; plain docstrings; match the plain-function style of
  `scripts/validation/vortex_detector.py`.
- Do NOT modify any existing file.

## Acceptance criteria

Tests in `tests/test_separated_flow_metrics.py`, all synthetic (no data files):

1. Attached flow: `tau_w = 1.0 + 0.1*sin(2*pi*x/5)` over x in [0, 10], 201 points →
   `separation_bubbles(...) == []`.
2. Single bubble: tau_w positive except a smooth negative region between x = 2.0 and
   x = 3.5 (≥ 5 samples each sign) → detected (x_sep, x_re) within 0.1 of (2.0, 3.5);
   `reattachment_length` ≈ 1.5.
3. Noise rejection: same profile but with ONE extra single-sample negative dip at x = 6.0 →
   still exactly one bubble; events unchanged.
4. `bubble_strength`: construct u with reverse flow peaking at -0.3 inside the window and
   forward flow outside → returns 0.3 (±1e-12).
5. `friction_velocity_from_gradient`: linear profile u = 2y → returns sqrt(2*nu) to 1e-10.
6. `detect_from_wall_data` returns the dict with consistent keys.

Run and record:

```
pytest tests/test_separated_flow_metrics.py -v        # all PASS
pytest tests/ -q                                      # existing suite still green (some skips are normal)
```

## When done

Write `parallel-tasks/TASK-01-RESULT.md`: files created, the two pytest summary lines, and
any deviation from this brief (should be none).
