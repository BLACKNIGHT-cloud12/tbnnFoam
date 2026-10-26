"""Shared TBNN tensor algebra (numpy-only, torch-free).

Single source of truth for the invariant/basis construction and the
Cardano realizability projection, used by the training script
(tbnn_train.py), the legacy experiments under python/, and the
validation scripts under scripts/validation/. The C++ counterpart of
this math lives in src/tbnnEvaluator.H and is kept in lockstep by
tests/test_parity.py.
"""
import numpy as np


def cardano_eigvals(A):
    p = 0.5 * np.sum(A**2)
    q = float(np.linalg.det(A))
    if p < 1e-14:
        return np.array([0.0, 0.0, 0.0])
    m = 2.0 * np.sqrt(p / 3.0)
    arg = np.clip(3.0 * q / (p * m), -1.0, 1.0)
    theta = (1.0 / 3.0) * np.arccos(arg)
    l1 = m * np.cos(theta)
    l2 = m * np.cos(theta - 2.0 * np.pi / 3.0)
    l3 = m * np.cos(theta + 2.0 * np.pi / 3.0)
    return np.sort(np.array([l1, l2, l3]))[::-1]


def project_realizable(A):
    tr = np.trace(A) / 3.0
    A_dev = A - tr * np.eye(3)
    l = cardano_eigvals(A_dev)
    scale = 1.0
    if l[2] < -1.0 / 3.0:
        scale = min(scale, (1.0 / 3.0) / abs(l[2]))
    if l[0] > 2.0 / 3.0:
        scale = min(scale, (2.0 / 3.0) / l[0])
    return scale * A_dev


def comp_inv_and_bases(grad_U, omega):
    N = grad_U.shape[0]
    tau = np.minimum(1.0 / np.clip(omega, 0.0001, 1000000.0), 5.0)
    S = 0.5 * (grad_U + np.swapaxes(grad_U, 1, 2))
    W = 0.5 * (grad_U - np.swapaxes(grad_U, 1, 2))
    s = S * tau[:, None, None]
    w = W * tau[:, None, None]
    s2 = np.matmul(s, s)
    w2 = np.matmul(w, w)
    s3 = np.matmul(s2, s)
    w2s = np.matmul(w2, s)
    w2s2 = np.matmul(w2, s2)
    I1 = np.trace(s2, axis1=1, axis2=2)[:, None]
    I2 = np.trace(w2, axis1=1, axis2=2)[:, None]
    I3 = np.trace(s3, axis1=1, axis2=2)[:, None]
    I4 = np.trace(w2s, axis1=1, axis2=2)[:, None]
    I5 = np.trace(w2s2, axis1=1, axis2=2)[:, None]
    inv = np.hstack([I1, I2, I3, I4, I5])
    I = np.eye(3)[None, :, :]
    T = np.zeros((N, 10, 3, 3))
    T[:, 0] = s
    T[:, 1] = np.matmul(s, w) - np.matmul(w, s)
    T[:, 2] = s2 - 1.0 / 3.0 * I1[:, :, None] * I
    T[:, 3] = w2 - 1.0 / 3.0 * I2[:, :, None] * I
    T[:, 4] = np.matmul(w, s2) - np.matmul(s2, w)
    T[:, 5] = np.matmul(w2, s) + np.matmul(s, w2) - 2.0 / 3.0 * I4[:, :, None] * I
    T[:, 6] = np.matmul(np.matmul(w, s), w2) - np.matmul(np.matmul(w2, s), w)
    T[:, 7] = np.matmul(np.matmul(s, w), s2) - np.matmul(np.matmul(s2, w), s)
    T[:, 8] = np.matmul(w2, s2) + np.matmul(s2, w2) - 2.0 / 3.0 * I5[:, :, None] * I
    T[:, 9] = np.matmul(np.matmul(w, s2), w2) - np.matmul(np.matmul(w2, s2), w)
    return (inv, T)
