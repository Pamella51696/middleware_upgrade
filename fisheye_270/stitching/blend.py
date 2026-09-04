"""Masks, photometric gain/offset, and feather blending.

Progression: hard seam -> feather (default) -> multiband if seams remain.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def distance_feather_mask(valid: np.ndarray, blend_width: int) -> np.ndarray:
    """Weight = 1 inside, ramps to 0 over ``blend_width`` pixels from the rim."""
    mask = (valid > 0).astype(np.uint8)
    if mask.max() == 0:
        return np.zeros(mask.shape, np.float32)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    w = np.clip(dist / max(float(blend_width), 1.0), 0.0, 1.0)
    w[mask == 0] = 0.0
    return w.astype(np.float32)


def hard_mask(valid: np.ndarray) -> np.ndarray:
    return (valid > 0).astype(np.float32)


def estimate_gain_offset(
    src: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    max_gain: float = 2.5,
    max_offset: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel I' = gain * I + offset using overlap mean/std."""
    m = mask > 0
    if m.sum() < 80:
        return np.ones(3, np.float32), np.zeros(3, np.float32)
    gains = []
    offsets = []
    for c in range(3):
        s = src[:, :, c][m].astype(np.float32)
        r = ref[:, :, c][m].astype(np.float32)
        ss, rs = float(s.std()) + 1e-3, float(r.std()) + 1e-3
        g = np.clip(rs / ss, 1.0 / max_gain, max_gain)
        o = np.clip(float(r.mean()) - g * float(s.mean()), -max_offset, max_offset)
        gains.append(g)
        offsets.append(o)
    return np.array(gains, np.float32), np.array(offsets, np.float32)


def apply_gain_offset(img: np.ndarray, gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32) * gain.reshape(1, 1, 3) + offset.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def feather_composite(
    layers: List[np.ndarray],
    weights: List[np.ndarray],
) -> np.ndarray:
    h, w = layers[0].shape[:2]
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for img, wt in zip(layers, weights):
        ww = wt.astype(np.float32)
        acc += img.astype(np.float32) * ww[..., None]
        wsum += ww
    wsum = np.maximum(wsum, 1e-6)
    return np.clip(acc / wsum[..., None], 0, 255).astype(np.uint8)


def seam_visualization(weights: Dict[str, np.ndarray], colors: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    ids = list(weights.keys())
    h, w = weights[ids[0]].shape[:2]
    vis = np.zeros((h, w, 3), np.uint8)
    stack = np.stack([weights[i] for i in ids], axis=-1)
    winner = np.argmax(stack, axis=-1)
    for idx, cid in enumerate(ids):
        vis[winner == idx] = colors.get(cid, (200, 200, 200))
    return vis


CAMERA_COLORS = {
    "FRONT": (80, 180, 255),
    "LEFT": (80, 220, 120),
    "RIGHT": (255, 180, 80),
    "REAR": (200, 120, 255),
}
