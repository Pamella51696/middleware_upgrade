"""Build debug mosaics for each development view (no GUI required)."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from ..runtime.processor import PipelineResult, Processor
from ..stitching.blend import CAMERA_COLORS


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def side_by_side(a: np.ndarray, b: np.ndarray, la: str, lb: str) -> np.ndarray:
    def fit(im):
        h, w = im.shape[:2]
        scale = 480 / max(h, 1)
        return cv2.resize(im, (int(w * scale), 480))
    a, b = fit(a), fit(b)
    return np.hstack([_caption(a, la), _caption(b, lb)])


def mosaic_from_result(proc: Processor, result: PipelineResult, view: str) -> np.ndarray:
    if view == "raw":
        return _caption(proc.label_quad(result.raw), "View 1 — RAW")
    if view == "undistorted":
        return _caption(proc.label_quad(result.undistorted), "View 2 — PERSPECTIVE UNDISTORT (debug)")
    if view == "projected":
        tiles = {k: v for k, v in result.projected.items()}
        return _caption(proc.label_quad(tiles), f"View 3 — {proc.spec.model.upper()} PROJECTION")
    if view == "overlap":
        return overlap_view(result)
    if view == "stitch":
        img = result.panorama if result.panorama is not None else np.zeros((200, 400, 3), np.uint8)
        miss = ",".join(result.missing) or "none"
        return _caption(img, f"View 7 — STITCH fov={proc.spec.fov_h_deg:.0f} missing={miss}")
    if view == "seams":
        img = result.seams if result.seams is not None else np.zeros((200, 400, 3), np.uint8)
        return _caption(img, "View 8 — SEAM WINNER")
    if view == "compare_proj":
        return compare_projections(proc, result)
    return proc.label_quad(result.raw)


def overlap_view(result: PipelineResult) -> np.ndarray:
    pairs = [("LEFT", "FRONT"), ("FRONT", "RIGHT")]
    rows = []
    for a, b in pairs:
        ia, ib = result.projected.get(a), result.projected.get(b)
        if ia is None or ib is None:
            continue
        blend = cv2.addWeighted(ia, 0.5, ib, 0.5, 0)
        tint = blend.copy()
        tint[:, :, 1] = cv2.addWeighted(ia[:, :, 1], 0.7, tint[:, :, 1], 0.3, 0)
        rows.append(_caption(cv2.resize(tint, (960, 270)), f"View 4 — overlap {a} + {b}"))
    if not rows:
        return np.zeros((270, 960, 3), np.uint8)
    return np.vstack(rows)


def compare_projections(proc: Processor, result: PipelineResult) -> np.ndarray:
    cid = "FRONT"
    raw = result.raw.get(cid)
    if raw is None:
        return np.zeros((240, 720, 3), np.uint8)
    panels = []
    for name in ("perspective", "cylindrical", "spherical"):
        if name == "perspective":
            img = result.undistorted.get(cid, raw)
        else:
            img, _ = proc.project(cid, raw, model_name=name)
        small = cv2.resize(img, (640, 180))
        panels.append(_caption(small, name))
    return np.vstack(panels)


def legend() -> np.ndarray:
    bar = np.zeros((40, 640, 3), np.uint8)
    x = 10
    for name, col in CAMERA_COLORS.items():
        cv2.rectangle(bar, (x, 8), (x + 24, 32), col, -1)
        cv2.putText(bar, name, (x + 30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x += 150
    return bar
