"""Synthetic overlapping fisheye cameras for tests (no vehicle required).

A cylindrical world texture is sampled by four virtual Kannala-Brandt cameras.
This is ESTIMATED geometry used only as a stand-in until real FRONT/LEFT/RIGHT/REAR
feeds are provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from ..calibration.approximate_intrinsics import estimate_intrinsics
from ..calibration.camera_model import CameraModel, Pose
from ..calibration.fisheye_model import project_fisheye


def make_world(width: int = 4096, height: int = 1024) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    # sky / ground
    img[: height // 2] = (210, 170, 90)
    img[height // 2 :] = (50, 50, 50)
    horizon = height // 2
    cv2.line(img, (0, horizon), (width, horizon), (240, 240, 240), 3)
    # 10° numbered columns + colored buildings
    for deg in range(-180, 180, 10):
        x = int((deg + 180) / 360.0 * width)
        color = (40 + (deg + 180) * 2 % 180, 80, 200 - (deg + 180) % 120)
        if deg % 30 == 0:
            cv2.rectangle(img, (x - 18, horizon - 180), (x + 18, horizon), color, -1)
            cv2.putText(img, f"{deg}", (x - 24, horizon - 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(img, (x, 0), (x, height), (30, 30, 30), 1)
    # road dashes
    for x in range(0, width, 40):
        cv2.rectangle(img, (x, horizon + 80), (x + 20, horizon + 92), (200, 200, 200), -1)
    cv2.putText(img, "FRONT 0 deg", (width // 2 - 80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


def _unproject_fisheye(u: np.ndarray, v: np.ndarray, cam: CameraModel) -> np.ndarray:
    intr = cam.intrinsics
    a = (u - intr.cx) / max(intr.fx, 1e-6)
    b = (v - intr.cy) / max(intr.fy, 1e-6)
    r = np.hypot(a, b)
    theta_d = r
    # invert theta_d = theta (1 + k1 th^2 + ...) with 4 Newton steps
    theta = theta_d.copy()
    for _ in range(4):
        t2 = theta * theta
        k = 1.0 + intr.k1 * t2 + intr.k2 * t2 * t2 + intr.k3 * t2 ** 3 + intr.k4 * t2 ** 4
        theta = np.where(k > 1e-8, theta_d / k, theta)
    z = np.cos(theta)
    s = np.sin(theta)
    scale = np.where(r > 1e-8, s / np.maximum(r, 1e-8), 1.0)
    x = a * scale
    y = b * scale
    x = np.where(r <= 1e-8, 0.0, x)
    y = np.where(r <= 1e-8, 0.0, y)
    rays_c = np.stack([x, y, z], axis=-1)
    R = cam.rotation_camera_to_vehicle()
    return rays_c @ R.T  # camera -> vehicle


def render_fisheye(world: np.ndarray, cam: CameraModel) -> np.ndarray:
    h, w = cam.intrinsics.height, cam.intrinsics.width
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    rays_v = _unproject_fisheye(uu, vv, cam)
    theta = np.arctan2(rays_v[..., 0], rays_v[..., 2])
    hyp = np.hypot(rays_v[..., 0], rays_v[..., 2])
    phi = np.arctan2(rays_v[..., 1], np.maximum(hyp, 1e-8))
    wh, ww = world.shape[:2]
    tx = (theta + np.pi) / (2 * np.pi) * (ww - 1)
    # phi ~ [-fov_v/2, fov_v/2] maps to world height; use +/- 50 deg
    ty = (phi + np.deg2rad(50.0)) / np.deg2rad(100.0) * (wh - 1)
    mapx = tx.astype(np.float32)
    mapy = np.clip(ty, 0, wh - 1).astype(np.float32)
    return cv2.remap(world, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def default_cameras(width: int = 640, height: int = 480, fov: float = 190.0) -> Dict[str, CameraModel]:
    yaws = {"FRONT": 0.0, "LEFT": -90.0, "RIGHT": 90.0, "REAR": 180.0}
    cams = {}
    for cid, yaw in yaws.items():
        # LEFT gets a small extra k1 so Mode A vs generating model is not identical.
        k1 = 0.0
        intr = estimate_intrinsics(width, height, fov, k1=k1)
        cams[cid] = CameraModel(
            camera_id=cid,
            intrinsics=intr,
            pose=Pose(yaw_deg=yaw, pitch_deg=0.0),
        )
    return cams


def synthetic_frames(
    n: int = 8, width: int = 640, height: int = 480
) -> Tuple[Dict[str, list], np.ndarray]:
    world = make_world()
    cams = default_cameras(width, height)
    frames: Dict[str, list] = {k: [] for k in cams}
    for i in range(n):
        # slow pan of the world to create motion without breaking overlap
        shift = int(i * 6)
        moved = np.roll(world, shift, axis=1)
        for cid, cam in cams.items():
            img = render_fisheye(moved, cam)
            if cid == "RIGHT":
                img = np.clip(img.astype(np.float32) * 1.15 + 8, 0, 255).astype(np.uint8)
            frames[cid].append(img)
    return frames, world


def write_videos(out_dir: Path, frames: Dict[str, list], fps: float = 15.0) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cid, seq in frames.items():
        path = out_dir / f"{cid.lower()}.mp4"
        h, w = seq[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        for im in seq:
            wr.write(im)
        wr.release()
        paths[cid] = path
    return paths


def roundtrip_error(cam: CameraModel, n: int = 400) -> float:
    """Sanity: project random rays and check they land in-frame for on-axis samples."""
    rng = np.random.default_rng(0)
    phi = rng.uniform(-0.3, 0.3, n)
    theta = rng.uniform(-0.4, 0.4, n)
    rays = np.stack([np.sin(theta), np.tan(phi), np.cos(theta)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    uv, valid = project_fisheye(rays, cam.intrinsics)
    return float(valid.mean())
