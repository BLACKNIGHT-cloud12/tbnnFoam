import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tbnn_train import comp_inv_and_bases, cardano_eigvals, project_realizable

def test_basis_trace_freedom():
    np.random.seed(42)
    grad_U = np.random.randn(20, 3, 3)
    for i in range(20):
        grad_U[i] -= (np.trace(grad_U[i]) / 3.0) * np.eye(3)
    omega = np.random.uniform(1.0, 100.0, size=20)
    inv, T = comp_inv_and_bases(grad_U, omega)
    trace_free_indices = list(range(10))
    for idx in trace_free_indices:
        traces = np.trace(T[:, idx, :, :], axis1=1, axis2=2)
        assert np.allclose(traces, 0.0, atol=1e-12)

def test_basis_symmetry():
    np.random.seed(42)
    grad_U = np.random.randn(20, 3, 3)
    omega = np.random.uniform(1.0, 100.0, size=20)
    inv, T = comp_inv_and_bases(grad_U, omega)
    for idx in range(10):
        t_base = T[:, idx, :, :]
        t_trans = np.swapaxes(t_base, 1, 2)
        assert np.allclose(t_base, t_trans, atol=1e-12)

def test_cardano_eigenvalue_accuracy():
    np.random.seed(123)
    for _ in range(50):
        mat = np.random.randn(3, 3)
        symm = 0.5 * (mat + mat.T)
        symm -= (np.trace(symm) / 3.0) * np.eye(3)
        eig_numpy = np.sort(np.linalg.eigvalsh(symm))[::-1]
        eig_cardano = cardano_eigvals(symm)
        assert np.allclose(eig_numpy, eig_cardano, atol=1e-12)

def test_realizability_bounds():
    np.random.seed(999)
    for _ in range(50):
        diag_violating = np.random.uniform(-1.5, 1.5, size=3)
        diag_violating -= np.mean(diag_violating)
        R, _ = np.linalg.qr(np.random.randn(3, 3))
        A_bad = R @ np.diag(diag_violating) @ R.T
        A_proj = project_realizable(A_bad)
        eigs = np.sort(np.linalg.eigvalsh(A_proj))[::-1]
        assert eigs[2] >= -1.0 / 3.0 - 1e-12
        assert eigs[0] <= 2.0 / 3.0 + 1e-12
        assert abs(np.trace(A_proj)) <= 1e-12
        k = 1.5
        R_stress = 2.0 * k * (A_proj + (1.0 / 3.0) * np.eye(3))
        r_eigs = np.linalg.eigvalsh(R_stress)
        assert np.all(r_eigs >= -1e-12)

def test_evaluator_parity():
    grad_U = np.array([[[0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    omega = np.array([20.0])
    inv, T = comp_inv_and_bases(grad_U, omega)
    tau = 1.0 / 20.0
    S12 = 0.5 * 10.0 * tau
    W12 = 0.5 * 10.0 * tau
    assert np.isclose(inv[0, 0], 2.0 * S12**2)
    assert np.isclose(inv[0, 1], -2.0 * W12**2)
    assert np.isclose(T[0, 0, 0, 1], S12)
    assert np.isclose(T[0, 1, 0, 0], -2.0 * S12 * W12)

def test_torchscript_parity():
    import torch
    from tbnn_train import TensorBasisNeuralNetwork
    model_pt = Path(__file__).resolve().parent.parent / 'tbnn_model.pt'
    weights_pth = Path(__file__).resolve().parent.parent / 'tbnn_weights.pth'
    if not model_pt.exists() or not weights_pth.exists():
        pytest.skip("Model files not found")
    py_model = TensorBasisNeuralNetwork()
    py_model.load_state_dict(torch.load(weights_pth, map_location='cpu'))
    py_model.eval()
    ts_model = torch.jit.load(str(model_pt))
    ts_model.eval()
    np.random.seed(42)
    inv_np = np.random.uniform(-5.0, 5.0, size=(10, 5)).astype(np.float32)
    bases_np = np.random.uniform(-2.0, 2.0, size=(10, 10, 3, 3)).astype(np.float32)
    inv_t = torch.tensor(inv_np)
    bases_t = torch.tensor(bases_np)
    with torch.no_grad():
        out_py = py_model(inv_t, bases_t)
        out_ts = ts_model(inv_t, bases_t)
    assert torch.allclose(out_py, out_ts, atol=1e-5)

def test_json_weights_parity():
    import json
    weights_json = Path(__file__).resolve().parent.parent / 'export_weights' / 'weights.json'
    if not weights_json.exists():
        pytest.skip("weights.json not found")
    with open(weights_json, 'r') as f:
        data = json.load(f)
    assert '0.weight' in data and len(data['0.weight']) == 30 and len(data['0.weight'][0]) == 5
    assert '0.bias' in data and len(data['0.bias']) == 30
    assert '2.weight' in data and len(data['2.weight']) == 30 and len(data['2.weight'][0]) == 30
    assert '2.bias' in data and len(data['2.bias']) == 30
    assert '4.weight' in data and len(data['4.weight']) == 30 and len(data['4.weight'][0]) == 30
    assert '4.bias' in data and len(data['4.bias']) == 30
    assert '6.weight' in data and len(data['6.weight']) == 10 and len(data['6.weight'][0]) == 30
    assert '6.bias' in data and len(data['6.bias']) == 10
    assert 'inv_mean' in data and len(data['inv_mean']) == 5
    assert 'inv_std' in data and len(data['inv_std']) == 5

if __name__ == '__main__':
    test_basis_trace_freedom()
    test_basis_symmetry()
    test_cardano_eigenvalue_accuracy()
    test_realizability_bounds()
    test_evaluator_parity()
    test_torchscript_parity()
    test_json_weights_parity()
    print('[+] All parity and cross-implementation verification tests passed!')
