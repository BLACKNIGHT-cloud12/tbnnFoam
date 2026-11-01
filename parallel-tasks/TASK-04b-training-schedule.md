# TASK-04b — Rework: mini-batch training schedule for `tbnn_train_hills.py`

Precise rework order. The TASK-04 pipeline is verified correct — the ONLY problem is the
optimization schedule: full-batch 400 epochs = 400 gradient steps, and the model
underfits its own training data (train-band R² = 0.132 ≈ val-band R² = 0.140).
Fix the schedule. Change NOTHING else — targets, split, masks, outputs, and artifact
protection stay exactly as they are (they are verified).

## Rework (edit `tbnn_train_hills.py` ONLY)

1. **Mini-batch SGD loop** replacing the full-batch loop:
   - `--batch-size` CLI arg, default **8192** (~11 steps/epoch at 91,378 training points
     → ~4,417 optimizer steps for 400 epochs, at roughly the same wall time as before).
   - Reproducible shuffling: `rng = np.random.default_rng(42)`; per epoch, permute the
     training indices, iterate fixed-size batches (last partial batch included).
   - One `optimizer.zero_grad()` / forward / loss / backward / `step()` PER BATCH.
2. **Cosine learning-rate anneal** from `lr` (0.002) down to `0.1*lr` across the TOTAL
   step count: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps,
   eta_min=0.1*lr)`, stepped once per batch. (`total_steps = epochs * ceil(n_train/batch)`.)
3. **Logging**: every 50 epochs print the epoch, the mean training minibatch loss over
   that epoch, the current LR, and the validation loss (as before).
4. **Track and report convergence evidence**: record the best validation loss and the
   epoch it occurred; at the end print whether the best val loss occurred in the last
   25% of training (plateau evidence) or earlier.
5. Everything else — fluid mask, gradient scaling, invariants, targets, spatial split,
   output filenames, v1-artifact protection — MUST remain untouched.

## Acceptance criteria

1. Training runs end-to-end (expect ≤ ~30 min on CPU; if slower, report the measured
   per-epoch time — do not silently reduce epochs).
2. Final printed metrics table (same format as TASK-04) + the convergence-evidence line.
3. v1 artifact hashes UNCHANGED (tbnn_model.pt, tbnn_weights.pth, export_weights/weights.json).
4. `pytest tests/ -q --ignore=release` still 35 passed, 2 skipped.
5. No promised R² target — report the final numbers honestly. The supervisor judges
   whether the plateau is sufficient for the a posteriori run.

## When done

Update `parallel-tasks/TASK-04-RESULT.md` with a "TASK-04b rework" section: the new
metrics table, the convergence-evidence line, hashes, and the measured wall time.
