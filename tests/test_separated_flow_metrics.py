import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validation.separated_flow_metrics import (
    wall_shear_events,
    separation_bubbles,
    reattachment_length,
    bubble_strength,
    friction_velocity_from_gradient,
    detect_from_wall_data,
)


def _make_single_bubble_profile(n_points=201):
    """Generate x in [0, 10] and tau_w with a single separation bubble between 2.0 and 3.5."""
    x = np.linspace(0.0, 10.0, n_points)
    # Smooth parabola positive everywhere except between 2.0 and 3.5
    tau_w = (x - 2.0) * (x - 3.5)
    return x, tau_w


def test_acceptance_1_attached_flow():
    """Criterion 1: Attached flow has no separation bubbles."""
    x = np.linspace(0.0, 10.0, 201)
    tau_w = 1.0 + 0.1 * np.sin(2.0 * np.pi * x / 5.0)
    assert separation_bubbles(tau_w, x) == []
    assert wall_shear_events(tau_w, x) == []


def test_acceptance_2_single_bubble():
    """Criterion 2: Single bubble between 2.0 and 3.5 detected within 0.1 and L_re ~ 1.5."""
    x, tau_w = _make_single_bubble_profile(201)
    bubbles = separation_bubbles(tau_w, x)
    assert len(bubbles) == 1
    x_sep, x_re = bubbles[0]
    assert np.isclose(x_sep, 2.0, atol=0.1)
    assert np.isclose(x_re, 3.5, atol=0.1)
    l_re = reattachment_length(bubbles[0])
    assert np.isclose(l_re, 1.5, atol=0.1)


def test_acceptance_3_noise_rejection():
    """Criterion 3: Single-sample negative dip at x = 6.0 rejected, leaving bubble unchanged."""
    x, tau_w = _make_single_bubble_profile(201)
    events_clean = wall_shear_events(tau_w, x, min_persistence=2)
    bubbles_clean = separation_bubbles(tau_w, x, min_persistence=2)

    # Insert single negative sample at x = 6.0
    idx_dip = np.argmin(np.abs(x - 6.0))
    tau_w_noisy = tau_w.copy()
    tau_w_noisy[idx_dip] = -1.0

    events_noisy = wall_shear_events(tau_w_noisy, x, min_persistence=2)
    bubbles_noisy = separation_bubbles(tau_w_noisy, x, min_persistence=2)

    assert len(bubbles_noisy) == 1
    assert len(events_noisy) == len(events_clean)
    for (k1, x1), (k2, x2) in zip(events_clean, events_noisy):
        assert k1 == k2
        assert np.isclose(x1, x2, atol=1e-12)
    assert np.isclose(bubbles_noisy[0][0], bubbles_clean[0][0], atol=1e-12)
    assert np.isclose(bubbles_noisy[0][1], bubbles_clean[0][1], atol=1e-12)


def test_acceptance_4_bubble_strength():
    """Criterion 4: Reverse flow peaking at -0.3 inside window returns 0.3 (+/- 1e-12)."""
    x = np.linspace(0.0, 10.0, 101)
    y = np.linspace(0.0, 1.0, 21)
    u = np.ones((len(y), len(x))) * 2.0  # forward flow everywhere

    # Put a reverse-flow peak inside [2.0, 3.5]
    idx_x = np.where((x >= 2.0) & (x <= 3.5))[0]
    u[5, idx_x[len(idx_x) // 2]] = -0.3

    strength = bubble_strength(y, u, x, 2.0, 3.5)
    assert np.isclose(strength, 0.3, atol=1e-12)

    # When all u >= 0 in the window, strength is 0.0
    u_positive = np.ones((len(y), len(x))) * 0.5
    assert bubble_strength(y, u_positive, x, 2.0, 3.5) == 0.0


def test_acceptance_5_friction_velocity_linear_profile():
    """Criterion 5: Linear profile u = 2y returns sqrt(2*nu) to 1e-10."""
    nu = 1.5e-5
    # Test on uniform grid
    y_uniform = np.array([0.0, 0.01, 0.02])
    u_uniform = 2.0 * y_uniform
    u_tau_uniform = friction_velocity_from_gradient(y_uniform, u_uniform, nu)
    expected = np.sqrt(2.0 * nu)
    assert np.isclose(u_tau_uniform, expected, atol=1e-10)

    # Test on non-uniform grid
    y_nonuniform = np.array([0.0, 0.005, 0.02])
    u_nonuniform = 2.0 * y_nonuniform
    u_tau_nonuniform = friction_velocity_from_gradient(y_nonuniform, u_nonuniform, nu)
    assert np.isclose(u_tau_nonuniform, expected, atol=1e-10)


def test_acceptance_6_detect_from_wall_data_keys():
    """Criterion 6: detect_from_wall_data returns consistent dictionary keys."""
    x, tau_w = _make_single_bubble_profile(201)

    # Case 1: Wall shear stress only
    report_minimal = detect_from_wall_data(x, tau_w)
    expected_keys = {"bubbles", "reattachment_lengths", "bubble_strengths", "u_tau"}
    assert set(report_minimal.keys()) == expected_keys
    assert len(report_minimal["bubbles"]) == 1
    assert len(report_minimal["reattachment_lengths"]) == 1
    assert report_minimal["bubble_strengths"] == []
    assert report_minimal["u_tau"] is None

    # Case 2: Full fields provided
    y = np.linspace(0.0, 1.0, 21)
    u = np.ones((len(y), len(x))) * 1.5
    # Inject negative velocity in the bubble
    idx_bubble = (x >= 2.0) & (x <= 3.5)
    u[2, idx_bubble] = -0.25
    nu = 1.0e-5

    report_full = detect_from_wall_data(x, tau_w, y=y, u=u, nu=nu)
    assert set(report_full.keys()) == expected_keys
    assert len(report_full["bubbles"]) == 1
    assert len(report_full["reattachment_lengths"]) == 1
    assert len(report_full["bubble_strengths"]) == 1
    assert np.isclose(report_full["bubble_strengths"][0], 0.25, atol=1e-12)
    assert isinstance(report_full["u_tau"], float)


def test_boundary_noise_suppression():
    """Oscillating samples at the very start and end do not create spurious events."""
    x, tau_w = _make_single_bubble_profile(201)
    # Perturb the very first sample and very last sample
    tau_w[0] = -5.0  # first sample negative, followed by positive
    tau_w[-1] = -5.0  # last sample negative, preceded by positive

    events = wall_shear_events(tau_w, x, min_persistence=2)
    assert len(events) == 2
    assert events[0][0] == "separation"
    assert events[1][0] == "reattachment"
    assert np.isclose(events[0][1], 2.0, atol=0.1)
    assert np.isclose(events[1][1], 3.5, atol=0.1)


def test_non_uniform_x_streamwise_spacing():
    """Zero-crossing interpolation works with geometrically stretched non-uniform x."""
    # Stretched mesh towards 2.0 and 3.5
    xi = np.linspace(0.0, 1.0, 201)
    x = 10.0 * (xi ** 1.5)  # non-uniform x spacing
    tau_w = (x - 2.5) * (x - 4.5)
    bubbles = separation_bubbles(tau_w, x, min_persistence=2)
    assert len(bubbles) == 1
    assert np.isclose(bubbles[0][0], 2.5, atol=0.05)
    assert np.isclose(bubbles[0][1], 4.5, atol=0.05)
    assert np.isclose(reattachment_length(bubbles[0]), 2.0, atol=0.05)


def test_multiple_bubbles():
    """Detect multiple consecutive separation bubbles."""
    x = np.linspace(0.0, 10.0, 501)
    # Positive, then negative [1.0, 2.0], positive, then negative [5.0, 6.5], then positive
    tau_w = np.ones_like(x)
    tau_w[(x >= 1.0) & (x <= 2.0)] = -1.0
    tau_w[(x >= 5.0) & (x <= 6.5)] = -2.0

    bubbles = separation_bubbles(tau_w, x, min_persistence=2)
    assert len(bubbles) == 2
    assert np.isclose(bubbles[0][0], 1.0, atol=0.05)
    assert np.isclose(bubbles[0][1], 2.0, atol=0.05)
    assert np.isclose(bubbles[1][0], 5.0, atol=0.05)
    assert np.isclose(bubbles[1][1], 6.5, atol=0.05)


def test_validation_errors():
    """Verify appropriate validation error handling for improper inputs."""
    with pytest.raises(ValueError, match="same length"):
        wall_shear_events([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        wall_shear_events([1.0, 2.0], [2.0, 1.0])

    with pytest.raises(ValueError, match="at least 1"):
        wall_shear_events([1.0, 2.0], [1.0, 2.0], min_persistence=0)

    with pytest.raises(ValueError, match="at least 3 points"):
        friction_velocity_from_gradient([0.0, 1.0], [0.0, 1.0], 1e-5)

    with pytest.raises(ValueError, match="non-negative"):
        friction_velocity_from_gradient([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], -1.0)

    with pytest.raises(ValueError, match="2D array"):
        bubble_strength([0.0, 1.0], [1.0, 2.0], [0.0, 1.0], 0.0, 1.0)
