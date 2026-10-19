import numpy as np
from scipy import ndimage

def detect_vortex_cores(omega_x, y_1d, z_1d, min_separation=0.2, min_strength=0.04, min_area=4, wall_margin=0.08):
    Ny = len(y_1d)
    Nz = len(z_1d)
    Z_g, Y_g = np.meshgrid(z_1d, y_1d, indexing='ij')
    interior_mask = (np.abs(Y_g) <= (1.0 - wall_margin)) & (np.abs(Z_g) <= (1.0 - wall_margin))
    detected = []

    for sign, factor in [('acw', -1.0), ('cw', 1.0)]:
        vort_field = factor * omega_x
        binary_mask = (vort_field > min_strength) & interior_mask
        labeled, num_features = ndimage.label(binary_mask)
        candidates = []
        for feat_id in range(1, num_features + 1):
            comp_mask = (labeled == feat_id)
            comp_size = np.sum(comp_mask)
            if comp_size < min_area:
                continue
            masked_vort = vort_field * comp_mask
            max_idx = np.unravel_index(np.argmax(masked_vort), masked_vort.shape)
            peak_val = float(masked_vort[max_idx])
            candidates.append({
                'y': float(Y_g[max_idx]),
                'z': float(Z_g[max_idx]),
                'strength': peak_val * factor,
                'type': sign,
                'area': int(comp_size)
            })
        candidates.sort(key=lambda c: abs(c['strength']), reverse=True)
        filtered = []
        for cand in candidates:
            too_close = False
            for kept in filtered:
                dist = np.sqrt((cand['y'] - kept['y']) ** 2 + (cand['z'] - kept['z']) ** 2)
                if dist < min_separation:
                    too_close = True
                    break
            if not too_close:
                filtered.append(cand)
        detected.extend(filtered)
    return detected
