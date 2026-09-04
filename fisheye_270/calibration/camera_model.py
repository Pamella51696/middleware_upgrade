from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class Provenance(str, Enum):
    KNOWN = "known"
    ASSUMED = "assumed"
    ESTIMATED = "estimated"
    CALIBRATED = "calibrated"
    UNKNOWN = "unknown"


@dataclass
class Intrinsics:
    """OpenCV camera matrix K and Kannala-Brandt fisheye D.

    K = [[fx, 0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    D = [k1, k2, k3, k4]

    Mode A fills these from FOV + resolution (estimated).
    Mode B overwrites them from cv2.fisheye.calibrate() (calibrated).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    width: int = 0
    height: int = 0
    fov_horizontal_deg: Optional[float] = None
    K_source: Provenance = Provenance.ESTIMATED
    D_source: Provenance = Provenance.ESTIMATED

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def D(self) -> np.ndarray:
        return np.array([[self.k1], [self.k2], [self.k3], [self.k4]], dtype=np.float64)

    def as_opencv(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.K(), self.D()


@dataclass
class Pose:
    """Camera pose in the vehicle frame.

    Zero-parallax prototype: tx=ty=tz=0 (UNKNOWN / ASSUMED).
    Yaw places the camera on the panorama. Pitch/roll are small corrections.
    """

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    source: Provenance = Provenance.ASSUMED


@dataclass
class Alignment2D:
    """Similarity in panorama pixel space. Frozen at runtime."""

    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    source: Provenance = Provenance.ESTIMATED
    confidence: float = 0.0
    inliers: int = 0


@dataclass
class CameraModel:
    camera_id: str
    intrinsics: Intrinsics
    pose: Pose = field(default_factory=Pose)
    alignment: Alignment2D = field(default_factory=Alignment2D)
    enabled: bool = True

    def rotation_camera_to_vehicle(self) -> np.ndarray:
        """Columns are camera X,Y,Z expressed in the vehicle frame.

        Vehicle: X right, Y down, Z forward (OpenCV).
        Camera look (+Z_cam) = (sin yaw, 0, cos yaw) before pitch/roll.
        """
        yaw = np.deg2rad(self.pose.yaw_deg)
        pitch = np.deg2rad(self.pose.pitch_deg)
        roll = np.deg2rad(self.pose.roll_deg)

        look = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
        world_up_down = np.array([0.0, 1.0, 0.0])
        right = np.cross(world_up_down, look)
        n = np.linalg.norm(right)
        if n < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / n
        down = np.cross(look, right)
        down = down / (np.linalg.norm(down) + 1e-12)

        cr, sr = np.cos(roll), np.sin(roll)
        right_r = cr * right + sr * down
        down_r = -sr * right + cr * down
        return np.column_stack([right_r, down_r, look])

    def vehicle_to_camera(self, rays: np.ndarray) -> np.ndarray:
        R = self.rotation_camera_to_vehicle()
        return rays @ R
