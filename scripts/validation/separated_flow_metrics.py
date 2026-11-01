"""Separated-flow validation metrics module.

Provides algorithmic metrics for separated flows over periodic hills and
other geometries: wall shear stress sign-change events, separation bubbles,
reattachment length, recirculation bubble strength, and friction velocity.
"""

import numpy as np


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
    tau_w = np.asarray(tau_w, dtype=float)
    x = np.asarray(x, dtype=float)

    if tau_w.ndim != 1 or x.ndim != 1:
        raise ValueError("tau_w and x must be 1D arrays")
    if len(tau_w) != len(x):
        raise ValueError("tau_w and x must have the same length")
    if min_persistence < 1:
        raise ValueError("min_persistence must be at least 1")

    n = len(tau_w)
    if n < min_persistence:
        return []

    if np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be strictly increasing")

    signs = np.where(tau_w >= 0.0, 1, -1)

    # Establish initial confirmed state: first run of >= min_persistence identical signs
    current_sign = None
    start_idx = 0
    run_length = 0

    for i in range(n):
        if i == 0 or signs[i] == signs[i - 1]:
            run_length += 1
        else:
            run_length = 1

        if run_length >= min_persistence:
            current_sign = signs[i]
            start_idx = i + 1
            break

    if current_sign is None:
        return []

    events = []
    idx = start_idx

    while idx < n:
        if signs[idx] == current_sign:
            idx += 1
            continue

        # Found sample with opposite sign. Count consecutive opposite samples.
        opp_sign = signs[idx]
        run_start = idx
        while idx < n and signs[idx] == opp_sign:
            idx += 1
        opp_run = idx - run_start

        if opp_run >= min_persistence:
            # Confirmed sign change.
            # The crossing occurred between run_start - 1 and run_start.
            i_prev = run_start - 1
            i_curr = run_start
            x1, x2 = x[i_prev], x[i_curr]
            t1, t2 = tau_w[i_prev], tau_w[i_curr]

            if t2 == t1:
                x_event = 0.5 * (x1 + x2)
            else:
                x_event = x1 - t1 * (x2 - x1) / (t2 - t1)

            kind = "separation" if current_sign == 1 else "reattachment"
            events.append((kind, float(x_event)))
            current_sign = opp_sign
        # If opp_run < min_persistence, noise rejected; current_sign stays unchanged.
        # idx is already positioned at the end of the rejected run.

    return events


def separation_bubbles(tau_w, x, min_persistence=2):
    """Return list of (x_sep, x_re) tuples, one per separation bubble."""
    events = wall_shear_events(tau_w, x, min_persistence=min_persistence)
    bubbles = []
    i = 0
    while i < len(events):
        kind, x_val = events[i]
        if kind == "separation":
            if i + 1 < len(events) and events[i + 1][0] == "reattachment":
                bubbles.append((float(x_val), float(events[i + 1][1])))
                i += 2
                continue
        i += 1
    return bubbles


def reattachment_length(bubble):
    """x_re - x_sep for one (x_sep, x_re) tuple."""
    x_sep, x_re = bubble
    return float(x_re - x_sep)


def bubble_strength(y, u, x, x_sep, x_re):
    """Peak reverse-flow speed inside the bubble: max |min(u, 0)| over all y and
    x in [x_sep, x_re] (u is 2D with shape (n_y, n_x), increasing x along axis 1).
    Returns 0.0 if no negative u exists in the window."""
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)

    if u.ndim != 2:
        raise ValueError("u must be a 2D array with shape (len(y), len(x))")
    if u.shape != (len(y), len(x)):
        raise ValueError(
            f"u shape {u.shape} does not match (len(y)={len(y)}, len(x)={len(x)})"
        )

    x_min, x_max = min(x_sep, x_re), max(x_sep, x_re)
    mask = (x >= x_min) & (x <= x_max)
    if not np.any(mask):
        return 0.0

    u_window = u[:, mask]
    min_val = np.min(u_window)
    if min_val < 0.0:
        return float(-min_val)
    return 0.0


def friction_velocity_from_gradient(y_wall_layer, u_wall_layer, nu):
    """u_tau = sqrt(nu * max(du/dy|_wall, 0)) using second-order one-sided
    finite differences from the first three wall-normal samples.
    y_wall_layer: first 3 wall-normal coords; u_wall_layer: velocity at those coords."""
    y_wall_layer = np.asarray(y_wall_layer, dtype=float)
    u_wall_layer = np.asarray(u_wall_layer, dtype=float)

    if len(y_wall_layer) < 3 or len(u_wall_layer) < 3:
        raise ValueError("y_wall_layer and u_wall_layer must have at least 3 points")
    if nu < 0.0:
        raise ValueError("nu must be non-negative")

    y0, y1, y2 = y_wall_layer[:3]
    u0, u1, u2 = u_wall_layer[:3]

    h1 = y1 - y0
    h2 = y2 - y1
    H = y2 - y0

    if h1 <= 0.0 or h2 <= 0.0:
        raise ValueError("y coordinates must be strictly increasing away from the wall")

    # Second-order one-sided finite difference on non-uniform spacing:
    # u'(y0) = [ H^2 * (u1 - u0) - h1^2 * (u2 - u0) ] / (h1 * h2 * H)
    du_dy = (H**2 * (u1 - u0) - h1**2 * (u2 - u0)) / (h1 * h2 * H)
    return float(np.sqrt(nu * max(du_dy, 0.0)))


def detect_from_wall_data(x, tau_w, y=None, u=None, nu=None, min_persistence=2):
    """Convenience report. Returns dict:
    {"bubbles": [...], "reattachment_lengths": [...], "bubble_strengths": [...],
     "u_tau": float or None (computed when y/u/nu given, at x[0])}"""
    bubbles = separation_bubbles(tau_w, x, min_persistence=min_persistence)
    re_lengths = [reattachment_length(b) for b in bubbles]

    if y is not None and u is not None:
        strengths = [
            bubble_strength(y, u, x, b[0], b[1]) for b in bubbles
        ]
    else:
        strengths = []

    u_tau = None
    if y is not None and u is not None and nu is not None:
        y_arr = np.asarray(y, dtype=float)
        u_arr = np.asarray(u, dtype=float)
        if len(y_arr) >= 3 and u_arr.ndim == 2 and u_arr.shape[0] >= 3:
            u_tau = friction_velocity_from_gradient(y_arr[:3], u_arr[:3, 0], nu)

    return {
        "bubbles": bubbles,
        "reattachment_lengths": re_lengths,
        "bubble_strengths": strengths,
        "u_tau": u_tau,
    }
