"""Offline pairwise alignment in panorama space.

Homography is physically wrong here: cameras have a baseline (parallax),
the world is not a plane, and the source is fisheye. After cylindrical
projection, neighboring cameras mainly need a small similarity:
    translation + rotation + scale
with optional affine. Unconstrained homography overfits road texture.

Low-texture fallback: if inliers < threshold, keep the last stored
transform (or identity). Runtime never re-runs RANSAC unless configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..calibration.camera_model import Alignment2D, Provenance


@dataclass
class MatchDebug:
    pair: Tuple[str, str]
    keypoints_a: list
    keypoints_b: list
    matches: list
    inlier_mask: Optional[np.ndarray]
    transform: Alignment2D
    method: str
    overlap_error: float


def detect_orb(image: np.ndarray, nfeatures: int = 2000, mask: np.ndarray | None = None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    orb = cv2.ORB_create(nfeatures=nfeatures)
    m = None
    if mask is not None:
        m = (mask > 0).astype(np.uint8)
        if m.max() == 0:
            m = None
    return orb.detectAndCompute(gray, m)


def match_bf(desc_a, desc_b):
    if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def _similarity_from_affine(M: np.ndarray) -> Alignment2D:
    """M is 2x3: [a -b tx; b a ty] approximately for similarity."""
    a, b = float(M[0, 0]), float(M[1, 0])
    scale = float(np.hypot(a, b))
    rot = float(np.rad2deg(np.arctan2(b, a)))
    return Alignment2D(
        dx=float(M[0, 2]),
        dy=float(M[1, 2]),
        rotation_deg=rot,
        scale=scale if scale > 1e-4 else 1.0,
        source=Provenance.ESTIMATED,
    )


def estimate_transform(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    mode: str = "similarity",
    ransac_thresh: float = 4.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Alignment2D]:
    if len(pts_a) < 6 or len(pts_b) < 6:
        return None, None, Alignment2D(source=Provenance.ASSUMED, confidence=0.0)

    if mode == "homography":
        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_thresh)
        al = Alignment2D(source=Provenance.ESTIMATED, confidence=0.0)
        if mask is not None:
            al.inliers = int(mask.ravel().sum())
            al.confidence = al.inliers / max(len(pts_a), 1)
        return H, mask, al

    if mode in ("similarity", "translation"):
        M, mask = cv2.estimateAffinePartial2D(
            pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
        )
    else:
        M, mask = cv2.estimateAffine2D(
            pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
        )
    if M is None:
        return None, mask, Alignment2D(source=Provenance.ASSUMED, confidence=0.0)
    al = _similarity_from_affine(M)
    if mask is not None:
        al.inliers = int(mask.ravel().sum())
        al.confidence = al.inliers / max(len(pts_a), 1)
    if mode == "translation":
        al.rotation_deg = 0.0
        al.scale = 1.0
        al.dx = float(np.median(pts_b[:, 0] - pts_a[:, 0]))
        al.dy = float(np.median(pts_b[:, 1] - pts_a[:, 1]))
    return M, mask, al


def physically_reasonable(al: Alignment2D, limits: Dict) -> bool:
    if abs(al.dx) > float(limits.get("max_abs_dx_px", 120)):
        return False
    if abs(al.dy) > float(limits.get("max_abs_dy_px", 80)):
        return False
    if abs(al.rotation_deg) > float(limits.get("max_abs_rotation_deg", 8)):
        return False
    if abs(al.scale - 1.0) > float(limits.get("max_scale_deviation", 0.15)):
        return False
    return True


def overlap_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 50:
        return 1e9
    da = a.astype(np.float32)
    db = b.astype(np.float32)
    diff = np.abs(da - db).mean(axis=2)
    return float(diff[mask > 0].mean())


def photometric_refine(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask: np.ndarray,
    start: Alignment2D,
    max_iters: int = 25,
) -> Alignment2D:
    """Coordinate descent on overlap L1. Small steps; rejects explosions."""
    best = Alignment2D(
        dx=start.dx, dy=start.dy, rotation_deg=start.rotation_deg, scale=start.scale,
        source=Provenance.ESTIMATED, confidence=start.confidence, inliers=start.inliers,
    )
    h, w = img_a.shape[:2]
    cur = np.array([best.dx, best.dy, best.rotation_deg, best.scale], dtype=np.float64)
    steps = np.array([2.0, 2.0, 0.3, 0.01])
    best_err = _warped_error(img_a, img_b, mask, cur, w, h)
    for _ in range(max_iters):
        improved = False
        for i in range(4):
            for sgn in (-1.0, 1.0):
                trial = cur.copy()
                trial[i] += sgn * steps[i]
                err = _warped_error(img_a, img_b, mask, trial, w, h)
                if err + 1e-6 < best_err:
                    best_err = err
                    cur = trial
                    improved = True
        if not improved:
            steps *= 0.5
            if steps[0] < 0.25:
                break
    best.dx, best.dy, best.rotation_deg, best.scale = [float(x) for x in cur]
    return best


def _warped_error(img_a, img_b, mask, params, w, h) -> float:
    dx, dy, rot, scale = params
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    warped = cv2.warpAffine(img_a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = mask
    if m.shape[:2] != warped.shape[:2]:
        return 1e9
    return overlap_l1(warped, img_b, m)


def align_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask: np.ndarray,
    stitching_cfg: Dict,
    pair: Tuple[str, str] = ("A", "B"),
) -> MatchDebug:
    cfg = stitching_cfg.get("alignment") or {}
    kpa, da = detect_orb(img_a, int(cfg.get("nfeatures", 2000)), mask)
    kpb, db = detect_orb(img_b, int(cfg.get("nfeatures", 2000)), mask)
    matches = match_bf(da, db)
    method = "orb+ransac"
    al = Alignment2D(source=Provenance.ASSUMED, confidence=0.0)
    inlier_mask = None
    if len(matches) >= 8:
        pts_a = np.float32([kpa[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kpb[m.trainIdx].pt for m in matches])
        _M, inlier_mask, al = estimate_transform(
            pts_a, pts_b,
            mode=str(cfg.get("transform", "similarity")),
            ransac_thresh=float(cfg.get("ransac_reproj_threshold", 4.0)),
        )
    min_in = int(cfg.get("min_inliers", 25))
    min_ratio = float(cfg.get("min_inlier_ratio", 0.25))
    low = al.inliers < min_in or al.confidence < min_ratio or not physically_reasonable(al, cfg)
    if low:
        # phase correlation fallback (translation only)
        g1 = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = (mask > 0).astype(np.float32)
        g1 *= m
        g2 *= m
        ys, xs = np.where(mask > 0)
        if len(xs) > 50:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            shift, _resp = cv2.phaseCorrelate(g1[y0:y1, x0:x1], g2[y0:y1, x0:x1])
        else:
            shift, _resp = (0.0, 0.0), 0.0
        al = Alignment2D(dx=float(shift[0]), dy=float(shift[1]), source=Provenance.ESTIMATED, confidence=0.15)
        method = "phase-correlation-fallback"
        if not physically_reasonable(al, cfg):
            al = Alignment2D(source=Provenance.ASSUMED, confidence=0.0)
            method = "hold-identity"

    if str(cfg.get("refine", "photometric")) == "photometric" and al.confidence > 0:
        al = photometric_refine(img_a, img_b, mask, al, int(cfg.get("refine_max_iters", 25)))
        if not physically_reasonable(al, cfg):
            al.confidence = 0.0
            method = "rejected-unreasonable"

    err = overlap_l1(img_a, img_b, mask)
    return MatchDebug(
        pair=pair,
        keypoints_a=list(kpa or []),
        keypoints_b=list(kpb or []),
        matches=matches,
        inlier_mask=inlier_mask,
        transform=al,
        method=method,
        overlap_error=err,
    )


def draw_matches(img_a: np.ndarray, img_b: np.ndarray, dbg: MatchDebug) -> np.ndarray:
    mask = dbg.inlier_mask
    matches = dbg.matches
    if mask is not None and len(mask) == len(matches):
        inliers = [m for m, keep in zip(matches, mask.ravel()) if keep]
    else:
        inliers = matches
    vis_in = cv2.drawMatches(
        img_a, dbg.keypoints_a, img_b, dbg.keypoints_b, inliers[:80], None,
        matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis_in
