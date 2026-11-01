"""Validate the periodic-hills SST baseline against the published reference values.

Reads the converged time dir (wallShearStress vol field, Cx/Cy cell centres, U),
reconstructs the structured 300x120 mesh layout, computes wall-tangential tau
along the hill, runs the TASK-01 separated-flow metrics, and compares with
Voet & Laizet (2020) Table 2 (reference case, Re=5600):
    reattachment x/h:  DNS 4.148 | RANS SST 7.684
    bubble height x/h=2: DNS 0.53 | RANS 0.62 ; x/h=4: DNS 0.14 | RANS 0.47
    bubble length:       DNS 3.73 | RANS 7.452 (Cf < 1e-3 criterion)

Run:  python -I validate_baseline.py [case_dir]
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.validation.separated_flow_metrics import separation_bubbles, reattachment_length

CASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
TIME = "3000"
H = 0.028
RHO, UB = 1.225, 2.9714
NX, NY = 300, 120


def read_scalar_field(path):
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\((.*?)\);",
                  txt, re.S)
    if not m:
        m = re.search(r"internalField\s+uniform\s+([\S]+);", txt)
        return np.array([float(m.group(1))])
    vals = re.findall(r"([-\d.eE+]+)", m.group(2))
    return np.array([float(v) for v in vals])


def read_vector_field(path):
    txt = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\((.*?)\);",
                  txt, re.S)
    assert m, f"no nonuniform vector field in {path}"
    nums = re.findall(r"\(([-\d.eE+\s]+)\)", m.group(2))
    return np.array([[float(x) for x in v.split()] for v in nums])


def patch_vectors(path, patch):
    txt = path.read_text()
    m = re.search(patch + r"\s*\{(.*?)\n\s*\}", txt, re.S)
    assert m, f"patch {patch} not found in {path}"
    body = m.group(1)
    mv = re.search(r"value\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\((.*?)\);", body, re.S)
    if not mv:
        mv = re.search(r"value\s+uniform\s+\(([-\d.eE+\s]+)\);", body)
        assert mv, f"no value list for {patch} in {path}"
        return np.array([[float(x) for x in mv.group(1).split()]])
    nums = re.findall(r"\(([-\d.eE+\s]+)\)", mv.group(2))
    return np.array([[float(x) for x in v.split()] for v in nums])


# --- mesh layout (blockMesh: x-fastest, then y) ---
ccx = read_scalar_field(CASE / TIME / "Cx").reshape(NY, NX)
ccy = read_scalar_field(CASE / TIME / "Cy").reshape(NY, NX)
U = read_vector_field(CASE / TIME / "U")[:, 0].reshape(NY, NX)
assert np.all(np.diff(ccx[0]) > 0), "cell-centre x not increasing along axis 1"

xh = ccx / H                      # (NY, NX) in h units
yh = ccy / H

# --- wall shear stress on hillLower (patch face order = x order) ---
tau = patch_vectors(CASE / TIME / "wallShearStress", "hillLower")
assert tau.shape[0] == NX, f"expected {NX} hillLower faces, got {tau.shape[0]}"

contour = np.loadtxt(CASE.parent / "data/benchmark_dns/periodic_hills/hill_contour_canonical.dat")
face_xh = ccx[0]                  # bottom-row cell centres approximate face x
slope = np.interp(face_xh, contour[:, 0], contour[:, 1])
t_hat = np.column_stack([np.ones_like(slope), slope])
t_hat /= np.linalg.norm(t_hat, axis=1, keepdims=True)
tau_t = np.einsum("ij,ij->i", tau, t_hat)   # wall-tangential component (downstream +)

x_face_h = face_xh
bubbles = separation_bubbles(tau_t, x_face_h, min_persistence=3)

q = 0.5 * RHO * UB ** 2
cf = np.linalg.norm(tau, axis=1) / q

# bubble length via the paper's criterion (Cf < 1e-3 along the wall)
below = cf < 1e-3
runs, start = [], None
for i, b in enumerate(below):
    if b and start is None:
        start = i
    if not b and start is not None:
        runs.append((x_face_h[start], x_face_h[i - 1]))
        start = None
if start is not None:
    runs.append((x_face_h[start], x_face_h[-1]))
cf_len = max((b - a) for a, b in runs) if runs else 0.0

# bubble heights at x/h = 2 and 4 (y-extent of U<0 above the local wall)
def bubble_height(x_target):
    i = int(np.argmin(np.abs(xh[0] - x_target)))
    col_u = U[:, i]
    col_y = yh[:, i]
    y_wall = np.interp(xh[0][i], contour[:, 0], contour[:, 1])
    rev = col_u < 0
    return (col_y[rev].max() - y_wall) if rev.any() else 0.0

bh2 = bubble_height(2.0)
bh4 = bubble_height(4.0)

print(f"converged time dir: {TIME}")
print(f"reverse-flow cells (U<0): {int((U < 0).sum())}")
print(f"tau_t range on hill: [{tau_t.min():.3f}, {tau_t.max():.3f}] Pa")
print()
print(f"{'quantity':<28}{'ours (SST)':>12}{'paper SST':>12}{'paper DNS':>12}")
reatt = reattachment_length(bubbles[0]) + bubbles[0][0] if bubbles else float("nan")
print(f"{'reattachment x/h':<28}{reatt:>12.3f}{7.684:>12.3f}{4.148:>12.3f}")
print(f"{'bubble length (Cf<1e-3)':<28}{cf_len:>12.3f}{7.452:>12.3f}{3.73:>12.3f}")
print(f"{'bubble height @ x/h=2':<28}{bh2:>12.3f}{0.62:>12.3f}{0.53:>12.3f}")
print(f"{'bubble height @ x/h=4':<28}{bh4:>12.3f}{0.47:>12.3f}{0.14:>12.3f}")
print(f"\nbubbles detected: {bubbles}")
print(f"Cf min along hill: {cf.min():.2e} (paper criterion threshold 1e-3 -> tau {1e-3 * q:.2e} Pa)")
