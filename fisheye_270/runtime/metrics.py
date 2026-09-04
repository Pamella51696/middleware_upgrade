"""Quality metrics used by tests and the debug UI."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def alignment_pixel_error(pts_a: np.ndarray, pts_b: np.ndarray, M: Optional[np.ndarray]) -> float:
    if M is None or len(pts_a) == 0:
        return float("nan")
    ones = np.ones((len(pts_a), 1), np.float32)
    p = np.hstack([pts_a, ones])
    pred = (M @ p.T).T
    err = np.linalg.norm(pred - pts_b, axis=1)
    return float(np.median(err))


def photometric_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    m = mask > 0
    if m.sum() < 20:
        return {"l1": float("nan"), "mean_delta": float("nan")}
    d = a.astype(np.float32) - b.astype(np.float32)
    return {
        "l1": float(np.abs(d)[m].mean()),
        "mean_delta": float(d[m].mean()),
    }


def seam_error(panorama: np.ndarray, seam_x: int, band: int = 6) -> float:
    x0 = max(seam_x - band, 0)
    x1 = min(seam_x + band, panorama.shape[1] - 1)
    left = panorama[:, x0:seam_x].astype(np.float32)
    right = panorama[:, seam_x:x1].astype(np.float32)
    if left.size == 0 or right.size == 0:
        return float("nan")
    return float(np.abs(left.mean(axis=1) - right.mean(axis=1)).mean())


def temporal_stability(prev: Optional[np.ndarray], cur: np.ndarray) -> float:
    if prev is None or prev.shape != cur.shape:
        return 0.0
    return float(np.mean(np.abs(prev.astype(np.int16) - cur.astype(np.int16))))
