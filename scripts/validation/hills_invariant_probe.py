"""Distribution-shift probe: periodic-hills DNS invariants vs the duct training envelope.

Computes the 5 tensor-basis invariants on the hills training plane (physical units,
same pipeline the v2 trainer will use) and reports 1D marginal coverage against the
square-duct (Re_tau=150) training envelope published in README.md. The joint 5D
k-NN shift requires the duct DNS CSV (not redistributed) and is a later step.

Run: python -I scripts/validation/hills_invariant_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tbnn_tensor_math import comp_inv_and_bases
from scripts.convert_hills_dns import grad_xy
CSV = ROOT / "data/benchmark_dns/periodic_hills/periodic_hills_dns_Re5600_reference.csv"
UB, H = 2.9714, 0.028
SCALE = UB / H          # physical gradient scale: (U_b/h) [1/s]
NX, NY = 512, 257

# Duct training envelope (README "Operating Envelope" table, Re_tau = 150 DNS)
ENVELOPE = {
    "I1": (0.0, 0.933),
    "I2": (-0.932, 0.0),
    "I3": (-0.0349, 0.00426),
    "I4": (-0.00142, 0.0116),
    "I5": (-0.435, 0.0),
}

df = pd.read_csv(CSV)
x = df.x_over_h.to_numpy().reshape(NY, NX)
y = df.y_over_h.to_numpy().reshape(NY, NX)
U = df.U_over_Ub.to_numpy().reshape(NY, NX)
V = df.V_over_Ub.to_numpy().reshape(NY, NX)
W = df.W_over_Ub.to_numpy().reshape(NY, NX)
k = df.k.to_numpy().reshape(NY, NX)
om_nd = df.omega.to_numpy().reshape(NY, NX)

fluid = k > 1e-10
print(f"plane {NY}x{NX} = {NY*NX} pts | fluid (k>1e-10): {fluid.sum()} "
      f"| solid/masked excluded: {(~fluid).sum()}")
print(f"max |W|/U_b = {np.abs(W[fluid]).max():.3e} (2D flow check)")

dUdx, dUdy = grad_xy(U, x, y)
dVdx, dVdy = grad_xy(V, x, y)

N = NY * NX
grad_U = np.zeros((N, 3, 3))
grad_U[:, 0, 0] = (dUdx * SCALE).ravel()
grad_U[:, 0, 1] = (dUdy * SCALE).ravel()
grad_U[:, 1, 0] = (dVdx * SCALE).ravel()
grad_U[:, 1, 1] = (dVdy * SCALE).ravel()
omega_phys = (om_nd * SCALE).ravel()

inv, T = comp_inv_and_bases(grad_U, omega_phys)
tau = np.minimum(1.0 / np.clip(omega_phys, 1e-4, 1e6), 5.0)
names = ["I1", "I2", "I3", "I4", "I5"]
fmask = fluid.ravel()

print(f"\ntau: median {np.median(tau[fmask]):.4f} s | capped at 5 s: "
      f"{100.0*np.mean(tau[fmask] >= 5.0):.2f}% of fluid points")
print(f"\n{'inv':<5}{'hills range (fluid)':>28}{'duct envelope':>24}{'coverage':>10}")
for i, nm in enumerate(names):
    v = inv[fmask, i]
    lo, hi = ENVELOPE[nm]
    cov = 100.0 * np.mean((v >= lo) & (v <= hi))
    print(f"{nm:<5}{f'[{v.min():+.4f}, {v.max():+.4f}]':>28}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}{f'{cov:6.1f}%':>10}")
print(f"\nNOTE: coverage <100% is EXPECTED and is the honest distribution-shift "
      f"number — the duct model has never seen separation or streamline curvature.")

# Where do the out-of-envelope cells live?
print("\n--- out-of-envelope cell localization (any invariant outside duct range) ---")
outside = np.zeros(N, dtype=bool)
for i, nm in enumerate(names):
    lo, hi = ENVELOPE[nm]
    outside |= (inv[:, i] < lo) | (inv[:, i] > hi)
outside &= fmask
xh_f = x.ravel()[outside]
yh_f = y.ravel()[outside]
kw = k.ravel()[outside]
print(f"out-of-envelope cells: {outside.sum()} ({100.0*outside.sum()/fmask.sum():.1f}% of fluid)")
print(f"  y/h distribution: median {np.median(yh_f):.3f} | 90th pct {np.percentile(yh_f,90):.3f} | max {yh_f.max():.3f}")
print(f"  fraction with y/h < 0.05 (near-wall ring): {100.0*np.mean(yh_f < 0.05):.1f}%")
print(f"  fraction with y/h < 0.02: {100.0*np.mean(yh_f < 0.02):.1f}%")
print(f"  k/U_b2 on those cells: median {np.median(kw):.2e}")
print(f"  x/h span: [{xh_f.min():.2f}, {xh_f.max():.2f}]")
# tau cap sensitivity: coverage if tau were capped at 5*h/U_b instead of 5 s
tau_cap = np.minimum(1.0 / np.clip(omega_phys, 1e-4, 1e6), 5.0 * H / UB)
S = 0.5 * (grad_U + np.swapaxes(grad_U, 1, 2))
Wt = 0.5 * (grad_U - np.swapaxes(grad_U, 1, 2))
s2c = np.matmul(S * tau_cap[:, None, None], S * tau_cap[:, None, None])
I1_cap = np.trace(s2c, axis1=1, axis2=2)
print(f"  with tau capped at 5*h/U_b = {5*H/UB:.4f} s: I1 max = {I1_cap[fmask].max():.3f} "
      f"(vs {inv[fmask,0].max():.3e}), I1>0.933 fraction = "
      f"{100.0*np.mean(I1_cap[fmask] > 0.933):.2f}%")
