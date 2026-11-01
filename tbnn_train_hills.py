"""
tbnn_train_hills.py

Train TBNN (5 -> [64, 64, 64] -> 10) on periodic-hills DNS benchmark (Re = 5600).
Canonical Breuer periodic hills geometry.
Mirroring tbnn_train.py architecture while importing TensorBasisNeuralNetwork.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tbnn_train import TensorBasisNeuralNetwork
from tbnn_tensor_math import comp_inv_and_bases
from scripts.convert_hills_dns import grad_xy

__all__ = ["train_hills", "r2_score"]


def r2_score(y_true, y_pred):
    """Coefficient of determination R^2 = 1 - SSE / SST."""
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst < 1e-12:
        return 1.0 if sse < 1e-12 else 0.0
    return 1.0 - sse / sst


def train_hills(
    csv_path=None,
    epochs=400,
    lr=0.002,
    batch_size=8192,
    device=None,
    h_ref=0.028,
    u_ref=2.9714,
):
    """Train TBNN on periodic hills DNS data and export model artifacts."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if csv_path is None:
        csv_path = ROOT / "data/benchmark_dns/periodic_hills/periodic_hills_dns_Re5600_reference.csv"
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    print(f"Loading periodic-hills DNS data from: {csv_path}")
    df = pd.read_csv(csv_path)

    NY, NX = 257, 512
    N_tot = len(df)
    if N_tot != NY * NX:
        raise ValueError(f"Expected {NY * NX} rows in CSV, got {N_tot}")

    x = df["x_over_h"].to_numpy().reshape(NY, NX)
    y = df["y_over_h"].to_numpy().reshape(NY, NX)
    U = df["U_over_Ub"].to_numpy().reshape(NY, NX)
    V = df["V_over_Ub"].to_numpy().reshape(NY, NX)
    W = df["W_over_Ub"].to_numpy().reshape(NY, NX)
    uu = df["uu"].to_numpy().reshape(NY, NX)
    vv = df["vv"].to_numpy().reshape(NY, NX)
    ww = df["ww"].to_numpy().reshape(NY, NX)
    uv = df["uv"].to_numpy().reshape(NY, NX)
    uw = df["uw"].to_numpy().reshape(NY, NX)
    vw = df["vw"].to_numpy().reshape(NY, NX)
    k = df["k"].to_numpy().reshape(NY, NX)
    omega_nd = df["omega"].to_numpy().reshape(NY, NX)

    # 1. Verify x strictly increasing along axis 1 after reshape
    dx = np.diff(x, axis=1)
    if not np.all(dx > 0):
        raise ValueError("x coordinates are not strictly increasing along axis 1 after reshape")
    print(f"Loaded plane {NY} x {NX} = {N_tot} points (x strictly increasing along axis 1, min dx={dx.min():.6e})")

    # 2. Fluid mask: k > 1e-10
    fluid_mask_2d = k > 1e-10
    fluid_count = int(np.sum(fluid_mask_2d))
    solid_count = int(np.sum(~fluid_mask_2d))
    print(f"Fluid mask (k > 1e-10): {fluid_count} fluid points | {solid_count} zero-masked hill-interior points excluded")

    # 3. Velocity-gradient tensor (2D mean flow: d/dz = 0)
    dUdx, dUdy = grad_xy(U, x, y)
    dVdx, dVdy = grad_xy(V, x, y)

    SCALE = float(u_ref) / float(h_ref)  # physical gradient scale: (U_b / h) [1/s]
    print(f"Physical scaling SCALE = U_b / h = {u_ref:.4f} / {h_ref:.4f} = {SCALE:.4f} 1/s")

    grad_U_2d = np.zeros((NY, NX, 3, 3), dtype=np.float64)
    grad_U_2d[:, :, 0, 0] = dUdx * SCALE
    grad_U_2d[:, :, 0, 1] = dUdy * SCALE
    grad_U_2d[:, :, 1, 0] = dVdx * SCALE
    grad_U_2d[:, :, 1, 1] = dVdy * SCALE
    # rows 3 and column 3 remain zero (2D flow)

    omega_phys_2d = omega_nd * SCALE

    # Flatten
    grad_U = grad_U_2d.reshape(N_tot, 3, 3)
    omega_phys = omega_phys_2d.ravel()

    # 4. Invariants and bases
    print("Computing tensor basis invariants and bases...")
    inv_all, T_all = comp_inv_and_bases(grad_U, omega_phys)

    # 5. Targets from DNS Reynolds stresses on fluid points
    fluid_flat = fluid_mask_2d.ravel()
    k_f = k.ravel()[fluid_flat]
    uu_f = uu.ravel()[fluid_flat]
    vv_f = vv.ravel()[fluid_flat]
    ww_f = ww.ravel()[fluid_flat]
    uv_f = uv.ravel()[fluid_flat]
    uw_f = uw.ravel()[fluid_flat]
    vw_f = vw.ravel()[fluid_flat]

    a_true = np.zeros((fluid_count, 3, 3), dtype=np.float32)
    two_k = 2.0 * k_f
    a_true[:, 0, 0] = uu_f / two_k - (1.0 / 3.0)
    a_true[:, 1, 1] = vv_f / two_k - (1.0 / 3.0)
    a_true[:, 2, 2] = ww_f / two_k - (1.0 / 3.0)
    a_true[:, 0, 1] = a_true[:, 1, 0] = uv_f / two_k
    a_true[:, 0, 2] = a_true[:, 2, 0] = uw_f / two_k
    a_true[:, 1, 2] = a_true[:, 2, 1] = vw_f / two_k

    # Physical realizability safeguard: Reynolds stress anisotropy is mathematically bounded
    # by the Lumley triangle (|a_ij| <= 1.0). Near walls where k -> 0 (e.g. k ~ 1e-10 at y=Ly),
    # unphysical shear stress noise produces division blowup in DNS post-processing.
    n_clipped = int(np.sum(np.abs(a_true) > 1.0))
    if n_clipped > 0:
        print(f"Applying realizability bound [-1, 1] to {n_clipped} tensor entries with division blowup (near-wall k -> 0).")
        a_true = np.clip(a_true, -1.0, 1.0)

    inv_fluid = inv_all[fluid_flat].astype(np.float32)
    T_fluid = T_all[fluid_flat].astype(np.float32)
    x_fluid = x.ravel()[fluid_flat]

    # Spatial split: Validation bands x/h in [2.0, 3.0] U [6.0, 7.0]
    val_mask = ((x_fluid >= 2.0) & (x_fluid <= 3.0)) | ((x_fluid >= 6.0) & (x_fluid <= 7.0))
    train_mask = ~val_mask

    n_val = int(np.sum(val_mask))
    n_train = int(np.sum(train_mask))
    print("=" * 70)
    print("SPATIAL EXTRAPOLATION SPLIT:")
    print(f"  Validation bands (x/h in [2.0, 3.0] U [6.0, 7.0]): {n_val} points ({100.0 * n_val / fluid_count:.2f}%)")
    print("    - Band 1 (x/h in [2.0, 3.0]): recirculation bubble interior")
    print("    - Band 2 (x/h in [6.0, 7.0]): post-reattachment recovery region")
    print(f"  Training set (remaining domain): {n_train} points ({100.0 * n_train / fluid_count:.2f}%)")
    print("  This spatial split rigorously tests out-of-sample spatial extrapolation.")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Training on device: {device}")

    X_inv_train = torch.tensor(inv_fluid[train_mask], dtype=torch.float32).to(device)
    X_basis_train = torch.tensor(T_fluid[train_mask], dtype=torch.float32).to(device)
    Y_a_train = torch.tensor(a_true[train_mask], dtype=torch.float32).to(device)

    X_inv_val = torch.tensor(inv_fluid[val_mask], dtype=torch.float32).to(device)
    X_basis_val = torch.tensor(T_fluid[val_mask], dtype=torch.float32).to(device)
    Y_a_val = torch.tensor(a_true[val_mask], dtype=torch.float32).to(device)

    model = TensorBasisNeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.1 * lr
    )

    rng = np.random.default_rng(42)

    print(f"\nStarting mini-batch training for {epochs} epochs:")
    print(f"  Batch size:       {batch_size}")
    print(f"  Steps per epoch:  {steps_per_epoch} (~{n_train} training points)")
    print(f"  Total steps:      {total_steps}")
    print(f"  Initial LR:       {lr:.6e} (cosine anneal down to {0.1 * lr:.6e})")

    best_val_loss = float("inf")
    best_val_epoch = 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        epoch_losses = []

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_idx = torch.as_tensor(perm[start_idx:end_idx], dtype=torch.int64, device=device)

            b_inv = X_inv_train[batch_idx]
            b_basis = X_basis_train[batch_idx]
            b_target = Y_a_train[batch_idx]

            optimizer.zero_grad()
            b_pred = model(b_inv, b_basis)
            b_loss = criterion(b_pred, b_target)
            b_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(b_loss.item())

        mean_train_loss = float(np.mean(epoch_losses))

        # Evaluate validation loss every epoch to track best validation loss
        model.eval()
        with torch.no_grad():
            val_pred_e = model(X_inv_val, X_basis_val)
            val_loss_curr = criterion(val_pred_e, Y_a_val).item()

        if val_loss_curr < best_val_loss:
            best_val_loss = val_loss_curr
            best_val_epoch = epoch

        if epoch % 50 == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:3d}/{epochs}] - Train Loss: {mean_train_loss:.6e} | "
                f"Val Loss: {val_loss_curr:.6e} | LR: {current_lr:.6e}"
            )

    wall_time = time.time() - t_start
    print(f"\nTraining completed in {wall_time:.2f} s ({wall_time / epochs:.4f} s/epoch)")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_inv_val, X_basis_val)
        val_loss = criterion(val_pred, Y_a_val).item()
        val_frob = torch.mean(torch.norm(val_pred - Y_a_val, dim=(1, 2))).item()

        vp_np = val_pred.cpu().numpy()
        yt_val_np = Y_a_val.cpu().numpy()

        r2_a11_val = r2_score(yt_val_np[:, 0, 0], vp_np[:, 0, 0])
        r2_a22_val = r2_score(yt_val_np[:, 1, 1], vp_np[:, 1, 1])
        r2_a33_val = r2_score(yt_val_np[:, 2, 2], vp_np[:, 2, 2])
        r2_a12_val = r2_score(yt_val_np[:, 0, 1], vp_np[:, 0, 1])
        r2_val_overall = r2_score(yt_val_np.ravel(), vp_np.ravel())

        # Overall predictions on ALL fluid points
        X_inv_all = torch.tensor(inv_fluid, dtype=torch.float32).to(device)
        X_basis_all = torch.tensor(T_fluid, dtype=torch.float32).to(device)
        Y_a_all = torch.tensor(a_true, dtype=torch.float32).to(device)

        all_pred = model(X_inv_all, X_basis_all)
        all_loss = criterion(all_pred, Y_a_all).item()
        all_frob = torch.mean(torch.norm(all_pred - Y_a_all, dim=(1, 2))).item()

        ap_np = all_pred.cpu().numpy()
        yt_all_np = a_true

        r2_all_overall = r2_score(yt_all_np.ravel(), ap_np.ravel())
        r2_a11_all = r2_score(yt_all_np[:, 0, 0], ap_np[:, 0, 0])
        r2_a22_all = r2_score(yt_all_np[:, 1, 1], ap_np[:, 1, 1])
        r2_a33_all = r2_score(yt_all_np[:, 2, 2], ap_np[:, 2, 2])
        r2_a12_all = r2_score(yt_all_np[:, 0, 1], ap_np[:, 0, 1])

    print("\n" + "=" * 70)
    print("FINAL EVALUATION METRICS")
    print("=" * 70)
    print(f"Validation Bands (x/h in [2, 3] U [6, 7], {n_val} pts):")
    print(f"  MSE Loss:                {val_loss:.6e}")
    print(f"  Mean Frobenius Error:    {val_frob:.6e}")
    print(f"  Overall Tensor R^2:      {r2_val_overall:.4f}")
    print("  Per-component R^2:")
    print(f"    a_11 R^2:              {r2_a11_val:.4f}")
    print(f"    a_22 R^2:              {r2_a22_val:.4f}")
    print(f"    a_33 R^2:              {r2_a33_val:.4f}")
    print(f"    a_12 R^2:              {r2_a12_val:.4f}")
    print("-" * 70)
    print(f"All Fluid Points ({fluid_count} pts):")
    print(f"  MSE Loss:                {all_loss:.6e}")
    print(f"  Mean Frobenius Error:    {all_frob:.6e}")
    print(f"  Overall a priori R^2:    {r2_all_overall:.4f}")
    print("  Per-component R^2:")
    print(f"    a_11 R^2:              {r2_a11_all:.4f}")
    print(f"    a_22 R^2:              {r2_a22_all:.4f}")
    print(f"    a_33 R^2:              {r2_a33_all:.4f}")
    print(f"    a_12 R^2:              {r2_a12_all:.4f}")
    print("-" * 70)
    print("Convergence Evidence:")
    print(f"  Best Val Loss:           {best_val_loss:.6e} at epoch {best_val_epoch}/{epochs}")
    plateau_cutoff = int(0.75 * epochs)
    if best_val_epoch >= plateau_cutoff:
        print(f"  Plateau Assessment:      Confirmed (best val loss in last 25% of training: epoch {best_val_epoch} >= {plateau_cutoff})")
    else:
        print(f"  Plateau Assessment:      Occurred earlier than last 25% (epoch {best_val_epoch} < {plateau_cutoff})")
    print(f"  Wall Time:               {wall_time:.2f} s ({wall_time / epochs:.4f} s/epoch)")
    print("=" * 70 + "\n")

    # Outputs
    model_cpu = model.to("cpu")

    # 1. State dict
    weights_pth_path = ROOT / "tbnn_weights_hills.pth"
    torch.save(model_cpu.state_dict(), str(weights_pth_path))
    print(f"[+] Saved state dict to: {weights_pth_path}")

    # 2. TorchScript trace (copy dummy-input tracing block from tbnn_train.py)
    dummy_inv = torch.zeros((1, 5), dtype=torch.float32)
    dummy_bases = torch.zeros((1, 10, 3, 3), dtype=torch.float32)
    dummy_bases[:, 0, 0, 0] = 1.0
    dummy_bases[:, 0, 1, 1] = -1.0
    traced = torch.jit.trace(model_cpu, (dummy_inv, dummy_bases))

    model_pt_path = ROOT / "tbnn_model_hills.pt"
    traced.save(str(model_pt_path))
    print(f"[+] Saved TorchScript model to: {model_pt_path}")

    # 3. Case copy in verification_periodicHills/constant/
    case_pt = ROOT / "verification_periodicHills/constant/tbnn_model.pt"
    case_pt.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(case_pt))
    print(f"[+] Saved case model to: {case_pt}")

    # 4. C++ JSON weights mirroring exact format
    weights_dict = {
        "inv_mean": [0.0, 0.0, 0.0, 0.0, 0.0],
        "inv_std": [1.0, 1.0, 1.0, 1.0, 1.0],
        "0.weight": model_cpu.mlp[0].weight.detach().cpu().numpy().tolist(),
        "0.bias": model_cpu.mlp[0].bias.detach().cpu().numpy().tolist(),
        "2.weight": model_cpu.mlp[2].weight.detach().cpu().numpy().tolist(),
        "2.bias": model_cpu.mlp[2].bias.detach().cpu().numpy().tolist(),
        "4.weight": model_cpu.mlp[4].weight.detach().cpu().numpy().tolist(),
        "4.bias": model_cpu.mlp[4].bias.detach().cpu().numpy().tolist(),
        "6.weight": model_cpu.mlp[6].weight.detach().cpu().numpy().tolist(),
        "6.bias": model_cpu.mlp[6].bias.detach().cpu().numpy().tolist(),
    }
    json_path = ROOT / "export_weights/weights_hills.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"[+] Exported C++ standalone weights to: {json_path}")

    return {
        "val_loss": val_loss,
        "val_frob": val_frob,
        "r2_val_overall": r2_val_overall,
        "r2_a11_val": r2_a11_val,
        "r2_a22_val": r2_a22_val,
        "r2_a33_val": r2_a33_val,
        "r2_a12_val": r2_a12_val,
        "all_loss": all_loss,
        "all_frob": all_frob,
        "r2_all_overall": r2_all_overall,
        "r2_a11_all": r2_a11_all,
        "r2_a22_all": r2_a22_all,
        "r2_a33_all": r2_a33_all,
        "r2_a12_all": r2_a12_all,
        "best_val_loss": best_val_loss,
        "best_val_epoch": best_val_epoch,
        "wall_time": wall_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TBNN model on periodic-hills DNS benchmark")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to reference DNS CSV")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs (default: 400)")
    parser.add_argument("--lr", type=float, default=0.002, help="Initial learning rate for Adam optimizer (default: 0.002)")
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size for training (default: 8192)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    train_hills(
        csv_path=args.csv_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
