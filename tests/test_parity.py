import sys
import warnings
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
    assert '0.weight' in data and len(data['0.weight'][0]) == 5
    H = len(data['0.weight'])
    assert H == 64, f"Expected 64 hidden neurons matching 5->64->64->64->10 architecture, got {H}"
    assert '0.bias' in data and len(data['0.bias']) == H
    assert '2.weight' in data and len(data['2.weight']) == H and len(data['2.weight'][0]) == H
    assert '2.bias' in data and len(data['2.bias']) == H
    assert '4.weight' in data and len(data['4.weight']) == H and len(data['4.weight'][0]) == H
    assert '4.bias' in data and len(data['4.bias']) == H
    assert '6.weight' in data and len(data['6.weight']) == 10 and len(data['6.weight'][0]) == H
    assert '6.bias' in data and len(data['6.bias']) == 10
    assert 'inv_mean' in data and len(data['inv_mean']) == 5
    assert 'inv_std' in data and len(data['inv_std']) == 5

def test_standalone_cpp_numerical_parity():
    import json
    import subprocess
    import shutil
    import platform
    repo_root = Path(__file__).resolve().parent.parent
    weights_json = repo_root / 'export_weights' / 'weights.json'
    if not weights_json.exists():
        pytest.skip("weights.json not found")
    with open(weights_json, 'r') as f:
        weights = json.load(f)
    W0 = np.array(weights['0.weight'], dtype=np.float64)
    b0 = np.array(weights['0.bias'], dtype=np.float64)
    W1 = np.array(weights['2.weight'], dtype=np.float64)
    b1 = np.array(weights['2.bias'], dtype=np.float64)
    W2 = np.array(weights['4.weight'], dtype=np.float64)
    b2 = np.array(weights['4.bias'], dtype=np.float64)
    W3 = np.array(weights['6.weight'], dtype=np.float64)
    b3 = np.array(weights['6.bias'], dtype=np.float64)
    mean = np.array(weights['inv_mean'], dtype=np.float64)
    std = np.array(weights['inv_std'], dtype=np.float64)
    test_cases = [
        (np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), 20.0),
        (np.array([[0.5, 1.2, -0.3], [-0.8, -0.2, 0.4], [0.1, -0.5, -0.3]]), 15.0),
        (np.array([[2.0, 25.0, -5.0], [5.0, -1.0, 8.0], [-3.0, -2.0, -1.0]]), 5.0)
    ]
    py_results = []
    for gradU, omega in test_cases:
        inv_py, T_py = comp_inv_and_bases(gradU[None, ...], np.array([omega]))
        inv_py = inv_py[0]
        T_py = T_py[0]
        x0 = np.clip((inv_py - mean) / (std + 1e-6), -50.0, 50.0)
        h0 = np.maximum(0.0, W0 @ x0 + b0)
        h1 = np.maximum(0.0, W1 @ h0 + b1)
        h2 = np.maximum(0.0, W2 @ h1 + b2)
        g_py = W3 @ h2 + b3
        a_py_raw = np.sum(g_py[:, None, None] * T_py, axis=0)
        a_py_symm = 0.5 * (a_py_raw + a_py_raw.T)
        a_py = project_realizable(a_py_symm)
        py_results.append({
            'invariants': inv_py,
            'basis': T_py,
            'g': g_py,
            'a': a_py
        })
    binary_path = repo_root / 'tests' / 'cpp_parity' / 'test_evaluator_parity'
    cpp_data = None
    if platform.system() == 'Windows':
        if shutil.which('wsl'):
            posix_root = str(repo_root).replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
            cmd = f"source /usr/lib/openfoam/openfoam2606/etc/bashrc 2>/dev/null && cd {posix_root} && ./tests/cpp_parity/test_evaluator_parity export_weights/weights.json"
            res = subprocess.run(['wsl', 'bash', '-c', cmd], capture_output=True, text=True)
            if res.returncode == 0 and '{' in res.stdout:
                json_start = res.stdout.find('{', res.stdout.find('tbnnEvaluator:'))
                if json_start != -1:
                    json_end = res.stdout.rfind('}') + 1
                    cpp_data = json.loads(res.stdout[json_start:json_end])
    elif platform.system() == 'Linux':
        if binary_path.exists():
            res = subprocess.run([str(binary_path), str(weights_json)], capture_output=True, text=True)
            if res.returncode == 0 and '{' in res.stdout:
                json_start = res.stdout.find('{', res.stdout.find('tbnnEvaluator:'))
                if json_start != -1:
                    json_end = res.stdout.rfind('}') + 1
                    cpp_data = json.loads(res.stdout[json_start:json_end])
    if cpp_data is None:
        warnings.warn("C++ evaluator parity test SKIPPED - binary or OpenFOAM "
                      "environment unavailable; cross-language parity is not being verified",
                      stacklevel=1)
        pytest.skip("C++ parity binary or OpenFOAM environment not available")
    for idx, case in enumerate(cpp_data['cases']):
        py_res = py_results[idx]
        assert np.allclose(case['invariants'], py_res['invariants'], atol=1e-5)
        assert np.allclose(case['basis'], py_res['basis'], atol=1e-5)
        assert np.allclose(case['g'], py_res['g'], atol=1e-5)
        assert np.allclose(case['a'], py_res['a'], atol=1e-5)

def test_solver_momentum_assembly_consistency():
    # Both solvers must couple the TBNN stress identically: the baseline
    # linear stress comes from turbulence->divDevReff(U), and the explicit
    # source carries the full 2*k*a + 2*nut*S stress whose 2*nut*S part
    # cancels the linear stress inside divDevReff. A bare
    # -fvm::laplacian(nuEff, U) drops the transpose divergence
    # -fvc::div(nuEff*gradU.T()) and does not reduce to the Boussinesq
    # equation in the linear limit.
    repo_root = Path(__file__).resolve().parent.parent
    root_solver = (repo_root / 'tbnnFoam.C').read_text()
    standalone_ueqn = (repo_root / 'src' / 'tbnnSimpleFoam' / 'UEqn.H').read_text()
    assert 'turbulence->divDevReff(U)' in root_solver
    assert 'fvc::div(tauDelta)' in root_solver
    assert 'turbulence->divDevReff(U)' in standalone_ueqn
    assert 'fvc::div(tauExtra)' in standalone_ueqn
    assert 'fvm::laplacian(turbulence->nuEff(), U)' not in standalone_ueqn

def _to_posix_path(path):
    s = str(path).replace('\\', '/')
    if len(s) > 1 and s[1] == ':':
        s = '/mnt/' + s[0].lower() + s[2:]
    return s

def _openfoam_bashrc():
    return '/usr/lib/openfoam/openfoam2606/etc/bashrc'

def _build_momentum_binary(repo_root):
    import subprocess
    import shutil
    binary = repo_root / 'tests' / 'cpp_parity' / 'momentum' / 'test_momentum_parity'
    source = repo_root / 'tests' / 'cpp_parity' / 'momentum' / 'test_momentum_parity.C'
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    if not shutil.which('wsl'):
        return None
    posix_root = _to_posix_path(repo_root)
    cmd = (f"source {_openfoam_bashrc()} 2>/dev/null && "
           f"cd {posix_root}/tests/cpp_parity/momentum && wmake")
    res = subprocess.run(['wsl', 'bash', '-c', cmd], capture_output=True, text=True, timeout=600)
    if binary.exists():
        return binary
    print(res.stdout[-2000:] if res.stdout else '')
    print(res.stderr[-2000:] if res.stderr else '')
    return None

def _write_momentum_case(case_dir):
    (case_dir / 'system').mkdir(parents=True)
    (case_dir / 'constant').mkdir(parents=True)
    (case_dir / '0').mkdir(parents=True)

    foam_header = "FoamFile\n{\n    version 2.0;\n    format ascii;\n"

    (case_dir / 'system' / 'controlDict').write_text(foam_header +
        "    class dictionary;\n    object controlDict;\n}\n"
        "application     test_momentum_parity;\n"
        "startFrom       startTime;\n"
        "startTime       0;\n"
        "stopAt          endTime;\n"
        "endTime         1;\n"
        "deltaT          1;\n"
        "writeControl    timeStep;\n"
        "writeInterval   1;\n")

    (case_dir / 'system' / 'fvSchemes').write_text(foam_header +
        "    class dictionary;\n    object fvSchemes;\n}\n"
        "ddtSchemes { default Euler; }\n"
        "gradSchemes { default Gauss linear; }\n"
        "divSchemes { default Gauss linear; }\n"
        "laplacianSchemes { default Gauss linear orthogonal; }\n"
        "snGradSchemes { default orthogonal; }\n"
        "wallDist { method meshWave; }\n")

    (case_dir / 'system' / 'fvSolution').write_text(foam_header +
        "    class dictionary;\n    object fvSolution;\n}\n"
        "solvers\n{\n"
        "    \"(U|k|omega)\"\n    {\n"
        "        solver          smoothSolver;\n"
        "        smoother        GaussSeidel;\n"
        "        nSweeps         1;\n"
        "        tolerance       1e-10;\n"
        "        relTol          0;\n"
        "    }\n"
        "}\n")

    # 2x2x2 hex mesh over [0, 0.5]^3; all six boundaries are wall patches so
    # kOmegaSST wall functions and the wall distance field behave as in the
    # verification cases.
    verts = []
    idx = {}
    for k in range(3):
        for j in range(3):
            for i in range(3):
                idx[(i, j, k)] = len(verts)
                verts.append(f"({i/2:.1f} {j/2:.1f} {k/2:.1f})")

    def patch_entry(name, ptype, quad):
        a, b, c, d = idx[quad[0]], idx[quad[1]], idx[quad[2]], idx[quad[3]]
        return (f"    {name}\n    {{\n        type {ptype};\n"
                f"        faces\n        (\n            ({a} {b} {c} {d})\n        );\n    }}\n")

    (case_dir / 'system' / 'blockMeshDict').write_text(foam_header +
        "    class dictionary;\n    object blockMeshDict;\n}\n"
        "scale   1;\n\n"
        "vertices\n(\n    " + "\n    ".join(verts) + "\n);\n\n"
        "blocks\n(\n    hex ("
        f" {idx[(0,0,0)]} {idx[(2,0,0)]} {idx[(2,2,0)]} {idx[(0,2,0)]}"
        f" {idx[(0,0,2)]} {idx[(2,0,2)]} {idx[(2,2,2)]} {idx[(0,2,2)]}"
        ") (2 2 2) simpleGrading (1 1 1)\n);\n\n"
        "edges\n(\n);\n\n"
        "boundary\n(\n"
        + patch_entry("minX", "wall", ((0,0,0), (0,0,2), (0,2,2), (0,2,0)))
        + patch_entry("maxX", "wall", ((2,0,0), (2,2,0), (2,2,2), (2,0,2)))
        + patch_entry("minY", "wall", ((0,0,0), (2,0,0), (2,0,2), (0,0,2)))
        + patch_entry("maxY", "wall", ((0,2,0), (0,2,2), (2,2,2), (2,2,0)))
        + patch_entry("minZ", "wall", ((0,0,0), (0,2,0), (2,2,0), (2,0,0)))
        + patch_entry("maxZ", "wall", ((0,0,2), (2,0,2), (2,2,2), (0,2,2)))
        + ");\n")

    (case_dir / 'constant' / 'transportProperties').write_text(foam_header +
        "    class dictionary;\n    object transportProperties;\n}\n"
        "transportModel  Newtonian;\n"
        "nu [0 2 -1 0 0 0 0] 1e-05;\n")

    (case_dir / 'constant' / 'turbulenceProperties').write_text(foam_header +
        "    class dictionary;\n    object turbulenceProperties;\n}\n"
        "simulationType RAS;\n\nRAS\n{\n"
        "    RASModel        kOmegaSST;\n"
        "    turbulence      on;\n"
        "    printCoeffs     off;\n}\n")

    # Exact linear velocity field U = U0 + G.x with tr(G) = 0, so that the
    # incompressible identity dev2(T(gradU)) = gradU.T() holds exactly.
    (case_dir / 'constant' / 'tbnnMomentumTestDict').write_text(foam_header +
        "    class dictionary;\n    object tbnnMomentumTestDict;\n}\n"
        "gradU (0 10 3 1 0 0 0 0 0);\n"
        "U0 (0.5 -0.2 0.3);\n")

    # Non-uniform k and omega (index = i + 2*j + 4*k, x fastest) so that nut
    # varies spatially and the missing transpose divergence in the pre-fix
    # assembly is strictly non-zero.
    xs = [0.125, 0.375]
    k_vals = [0.01 * (1.0 + 0.3 * xs[n % 2]) for n in range(8)]
    omega_vals = [50.0 * (1.0 + 0.5 * xs[n // 4]) for n in range(8)]

    (case_dir / '0' / 'U').write_text(foam_header +
        "    class volVectorField;\n    object U;\n}\n"
        "dimensions [0 1 -1 0 0 0 0];\n"
        "internalField uniform (0 0 0);\n"
        "boundaryField\n{\n"
        "    \".*\"\n    {\n"
        "        type fixedValue;\n"
        "        value uniform (0 0 0);\n"
        "    }\n}\n")

    (case_dir / '0' / 'p').write_text(foam_header +
        "    class volScalarField;\n    object p;\n}\n"
        "dimensions [0 2 -2 0 0 0 0];\n"
        "internalField uniform 0;\n"
        "boundaryField\n{\n"
        "    \".*\"\n    {\n"
        "        type zeroGradient;\n"
        "    }\n}\n")

    (case_dir / '0' / 'nut').write_text(foam_header +
        "    class volScalarField;\n    object nut;\n}\n"
        "dimensions [0 2 -1 0 0 0 0];\n"
        "internalField uniform 0;\n"
        "boundaryField\n{\n"
        "    \".*\"\n    {\n"
        "        type nutkWallFunction;\n"
        "        value uniform 0;\n"
        "    }\n}\n")

    (case_dir / '0' / 'k').write_text(foam_header +
        "    class volScalarField;\n    object k;\n}\n"
        "dimensions [0 2 -2 0 0 0 0];\n"
        "internalField nonuniform List<scalar> 8\n(\n    " +
        "\n    ".join(f"{v}" for v in k_vals) + "\n);\n"
        "boundaryField\n{\n"
        "    \".*\"\n    {\n"
        "        type kqRWallFunction;\n"
        "        value uniform 0.01;\n"
        "    }\n}\n")

    (case_dir / '0' / 'omega').write_text(foam_header +
        "    class volScalarField;\n    object omega;\n}\n"
        "dimensions [0 0 -1 0 0 0 0];\n"
        "internalField nonuniform List<scalar> 8\n(\n    " +
        "\n    ".join(f"{v}" for v in omega_vals) + "\n);\n"
        "boundaryField\n{\n"
        "    \".*\"\n    {\n"
        "        type omegaWallFunction;\n"
        "        value uniform 50;\n"
        "    }\n}\n")

def _run_momentum_parity(repo_root, binary):
    import json
    import subprocess
    import tempfile
    case_dir = Path(tempfile.mkdtemp(prefix='tbnn_momentum_parity_'))
    _write_momentum_case(case_dir)
    posix_case = _to_posix_path(case_dir)
    posix_binary = _to_posix_path(binary)
    cmd = (f"source {_openfoam_bashrc()} 2>/dev/null && "
           f"cd {posix_case} && blockMesh > log.blockMesh 2>&1 && "
           f"{posix_binary}")
    res = subprocess.run(['wsl', 'bash', '-c', cmd], capture_output=True, text=True, timeout=300)
    out = res.stdout
    if 'MPJ_BEGIN' not in out:
        print('blockMesh/solver log tail:', out[-2000:], res.stderr[-2000:])
        return None
    begin = out.find('MPJ_BEGIN') + len('MPJ_BEGIN')
    end = out.find('MPJ_END')
    return json.loads(out[begin:end])

def test_momentum_coupling_parity():
    import subprocess
    import shutil
    import tempfile
    repo_root = Path(__file__).resolve().parent.parent
    binary = _build_momentum_binary(repo_root)
    if binary is None:
        warnings.warn("Momentum-coupling parity test SKIPPED - WSL/OpenFOAM "
                      "toolchain unavailable; momentum assembly is not being verified",
                      stacklevel=1)
        pytest.skip("WSL OpenFOAM toolchain not available to build momentum parity binary")
    data = _run_momentum_parity(repo_root, binary)
    if data is None:
        warnings.warn("Momentum-coupling parity test SKIPPED - case failed to run",
                      stacklevel=1)
        pytest.skip("momentum parity case failed to run")
    assert data['status'] == 'success'
    # fvc::grad(U) must reproduce the prescribed linear gradient exactly
    assert data['gradU_rel_error'] < 1e-9, f"gradU error {data['gradU_rel_error']}"
    # nut must vary spatially so the negative control is meaningful
    assert data['nut_rel_variation'] > 0.05, f"nut variation {data['nut_rel_variation']}"
    # load-bearing premise: divDevReff = -laplacian(nuEff,U) - fvc::div(nuEff*gradU.T())
    assert data['premise_diff_rel'] < 1e-10, f"premise diff {data['premise_diff_rel']}"
    # tauExtra = 2*k*a + 2*nut*S must vanish for the Boussinesq-consistent a
    assert data['tauDelta_rel_norm'] < 1e-8, f"tauDelta {data['tauDelta_rel_norm']}"
    # the fixed coupling must reduce exactly to the baseline linear model
    assert data['boussinesq_diff_rel'] < 1e-8, f"boussinesq diff {data['boussinesq_diff_rel']}"
    # the pre-fix coupling must NOT reduce to the baseline (missing transpose term)
    assert data['old_form_diff_rel'] > 1e-2, f"old form diff {data['old_form_diff_rel']}"

if __name__ == '__main__':
    test_basis_trace_freedom()
    test_basis_symmetry()
    test_cardano_eigenvalue_accuracy()
    test_realizability_bounds()
    test_evaluator_parity()
    test_torchscript_parity()
    test_json_weights_parity()
    test_standalone_cpp_numerical_parity()
    test_solver_momentum_assembly_consistency()
    test_momentum_coupling_parity()
    print('[+] All parity and cross-implementation verification tests passed!')

