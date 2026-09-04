"""OpenCV fisheye (Kannala-Brandt) projection and undistort LUTs.

cv2.fisheye.calibrate(object_points, image_points, image_size, K, D)
    -> RMS, K, D, rvecs, tvecs
    Requires a calibration target. Mode B only.

cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, R, P, balance)
    Chooses a rectilinear K' after undistortion.
    balance=0 crops to valid pixels; 1 keeps FOV (black borders).

cv2.fisheye.initUndistortRectifyMap(K, D, R, P, size, m1type)
    Builds mapX/mapY: dest rectilinear pixel -> source fisheye pixel.

cv2.remap(src, mapX, mapY, INTER_LINEAR)
    Runtime: one bilinear sample per output pixel. Maps are precomputed.

Why LUTs: a 1920x540 remap is a memory gather. Recomputing atan/polynomial
per frame is wasted work while K, D, and pose are static.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .camera_model import Intrinsics


def undistort_maps(
    intr: Intrinsics,
    dst_size: Tuple[int, int] | None = None,
    balance: float = 0.5,
    fov_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perspective (rectilinear) undistort maps. Debug / Phase 2 view only.

    A 190° fisheye cannot be shown as a single perspective image without
    either cropping FOV or stretching the rim to infinity. Do not stitch
    these frames for the 270° panorama — use cylindrical/spherical maps.
    """
    w, h = (dst_size if dst_size else (intr.width, intr.height))
    K, D = intr.as_opencv()
    R = np.eye(3, dtype=np.float64)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (intr.width, intr.height), R, balance=float(balance),
        new_size=(w, h), fov_scale=float(fov_scale),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, R, new_K, (w, h), cv2.CV_32FC1,
    )
    return map1, map2, new_K


def project_fisheye(rays_cam: np.ndarray, intr: Intrinsics) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-frame rays to fisheye pixels. Supports FOV > 180° (z < 0).

    theta = atan2(sqrt(x^2+y^2), z)
    theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
    u = fx * (theta_d * x / r) + cx
    """
    x = rays_cam[..., 0]
    y = rays_cam[..., 1]
    z = rays_cam[..., 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    scale = 1.0 + intr.k1 * t2 + intr.k2 * t4 + intr.k3 * t6 + intr.k4 * t8
    theta_d = theta * scale
    # rays behind the camera at exactly 180° have r~0 and theta=pi — invalid
    valid = (r > 1e-12) | (np.abs(theta) < 1e-8)
    inv_r = np.where(r > 1e-12, 1.0 / np.maximum(r, 1e-12), 0.0)
    u = intr.fx * theta_d * x * inv_r + intr.cx
    v = intr.fy * theta_d * y * inv_r + intr.cy
    u = np.where(r <= 1e-12, intr.cx, u)
    v = np.where(r <= 1e-12, intr.cy, v)
    in_image = (
        valid
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (v >= 0)
        & (u < intr.width)
        & (v < intr.height)
        & (theta < np.deg2rad(min(max((intr.fov_horizontal_deg or 190.0) * 0.55, 80.0), 170.0)))
    )
    return np.stack([u, v], axis=-1).astype(np.float32), in_image


def remap(frame: np.ndarray, mapx: np.ndarray, mapy: np.ndarray) -> np.ndarray:
    return cv2.remap(
        frame, mapx, mapy, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
