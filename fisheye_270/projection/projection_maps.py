"""Panoramic coordinate system (NOT bird's-eye).

Pixel (x, y) on the output canvas:
    theta = yaw_center - fov_h/2 + x * (fov_h / width)     # heading, rad
    phi   = (y - height/2) * (fov_v / height)              # pitch, rad
            (cylindrical: phi from linear angle;
             spherical:   phi is latitude)

Vehicle ray (OpenCV: X right, Y down, Z forward):
    dir = [sin(theta) * cos(phi),
           sin(phi),
           cos(theta) * cos(phi)]

Then:
    camera pixel -> (already on sensor)
    camera ray   <- inverse fisheye  (or forward: pano ray -> camera)
    angular dir  -> theta, phi in vehicle frame
    pano coord   -> x = (theta - theta0) / fov * width

Homography is not used as the primary warp: the four cameras do not
share a single plane, and fisheye rays are not perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from ..calibration.camera_model import CameraModel
from ..calibration.fisheye_model import project_fisheye, undistort_maps


@dataclass
class PanoramaSpec:
    width: int
    height: int
    fov_h_deg: float
    fov_v_deg: float
    yaw_center_deg: float = 0.0
    model: str = "cylindrical"

    @property
    def theta0(self) -> float:
        return np.deg2rad(self.yaw_center_deg - self.fov_h_deg / 2.0)

    @property
    def theta1(self) -> float:
        return np.deg2rad(self.yaw_center_deg + self.fov_h_deg / 2.0)


def _grid_angles(spec: PanoramaSpec) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(spec.width, dtype=np.float32)
    ys = np.arange(spec.height, dtype=np.float32)
    x, y = np.meshgrid(xs, ys)
    theta = spec.theta0 + (x / max(spec.width - 1, 1)) * (spec.theta1 - spec.theta0)
    # vertical: map y=0 to -fov_v/2 (up, negative Y in OpenCV is up on screen
    # so y=0 is top = negative phi if Y-down)
    phi = ((y / max(spec.height - 1, 1)) - 0.5) * np.deg2rad(spec.fov_v_deg)
    return theta, phi


def _rays_from_angles(theta: np.ndarray, phi: np.ndarray, model: str) -> np.ndarray:
    if model == "spherical":
        # equirectangular: phi is elevation from horizon
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(phi)
        dz = np.cos(theta) * np.cos(phi)
    elif model == "cylindrical":
        # keep vertical as angle (less polar stretch than equirect at top/bottom)
        dx = np.sin(theta)
        dy = np.tan(phi)
        dz = np.cos(theta)
        n = np.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / n, dy / n, dz / n
    else:
        raise ValueError(f"unknown panoramic model {model}")
    return np.stack([dx, dy, dz], axis=-1)


def _apply_alignment(mapx: np.ndarray, mapy: np.ndarray, cam: CameraModel) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the camera map after a 2D similarity in panorama space.

    Destination pano pixel p is pulled from source map at T^{-1} p so that
    the projected camera image is translated/rotated/scaled on the canvas.
    """
    a = cam.alignment
    if (
        abs(a.dx) < 1e-6
        and abs(a.dy) < 1e-6
        and abs(a.rotation_deg) < 1e-6
        and abs(a.scale - 1.0) < 1e-6
    ):
        return mapx, mapy

    h, w = mapx.shape[:2]
    angle = np.deg2rad(a.rotation_deg)
    s = max(float(a.scale), 1e-4)
    ca, sa = np.cos(angle), np.sin(angle)
    # inverse similarity: p_src = R(-) / s * (p - t)  with t around image center
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ys, xs = np.indices((h, w), dtype=np.float32)
    px = xs - cx - float(a.dx)
    py = ys - cy - float(a.dy)
    qx = (ca * px + sa * py) / s + cx
    qy = (-sa * px + ca * py) / s + cy
    mapx_i = cv2.remap(mapx, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mapy_i = cv2.remap(mapy, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return mapx_i, mapy_i


def fisheye_to_panorama_maps(
    cam: CameraModel,
    spec: PanoramaSpec,
    model: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute dest pano -> source fisheye maps plus a validity mask."""
    model = model or spec.model
    if model == "perspective":
        # Debug path: undistort to a camera-sized rectilinear image; caller
        # places it later. For a full canvas we still use cylindrical rays
        # through a *perspective* pinhole (invalid for >~120°). Kept for compare.
        map1, map2, _ = undistort_maps(cam.intrinsics)
        return map1, map2, np.ones(map1.shape, dtype=np.uint8) * 255

    theta, phi = _grid_angles(spec)
    rays_v = _rays_from_angles(theta, phi, model)
    rays_c = cam.vehicle_to_camera(rays_v)
    uv, valid = project_fisheye(rays_c, cam.intrinsics)
    mapx = uv[..., 0].copy()
    mapy = uv[..., 1].copy()
    mapx[~valid] = -1
    mapy[~valid] = -1
    mapx, mapy = _apply_alignment(mapx, mapy, cam)
    mask = (mapx >= 0) & (mapy >= 0) & (mapx < cam.intrinsics.width) & (mapy < cam.intrinsics.height)
    return mapx, mapy, mask.astype(np.uint8) * 255


class MapCache:
    """Rebuild LUTs only when camera geometry or the canvas changes."""

    def __init__(self) -> None:
        self._key: Dict[str, tuple] = {}
        self._maps: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(
        self, cam: CameraModel, spec: PanoramaSpec, model: str | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = model or spec.model
        key = (
            cam.camera_id,
            model,
            spec.width,
            spec.height,
            spec.fov_h_deg,
            spec.fov_v_deg,
            spec.yaw_center_deg,
            cam.intrinsics.fx,
            cam.intrinsics.fy,
            cam.intrinsics.cx,
            cam.intrinsics.cy,
            cam.intrinsics.k1,
            cam.intrinsics.k2,
            cam.intrinsics.k3,
            cam.intrinsics.k4,
            cam.intrinsics.width,
            cam.intrinsics.height,
            cam.pose.yaw_deg,
            cam.pose.pitch_deg,
            cam.pose.roll_deg,
            cam.alignment.dx,
            cam.alignment.dy,
            cam.alignment.rotation_deg,
            cam.alignment.scale,
        )
        if self._key.get(cam.camera_id) != key:
            self._maps[cam.camera_id] = fisheye_to_panorama_maps(cam, spec, model)
            self._key[cam.camera_id] = key
        return self._maps[cam.camera_id]
