"""Mode A: approximate K, D from image size and an assumed FOV.

Equidistant fisheye (OpenCV D = 0):
    r_pixels = f * theta
    (width / 2) = fx * (fov_h / 2)

This is NOT a substitute for cv2.fisheye.calibrate(). It produces a
visually plausible unwarp. Principal point is assumed at the image center.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .camera_model import Intrinsics, Provenance


def estimate_intrinsics(
    width: int,
    height: int,
    fov_horizontal_deg: float,
    fov_vertical_deg: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    k_source: str = "estimated",
    d_source: str = "estimated",
) -> Intrinsics:
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive (KNOWN from the first frame)")
    if fov_horizontal_deg <= 0:
        raise ValueError("fov_horizontal_deg must be positive")

    half_fov_h = np_radians(fov_horizontal_deg) / 2.0
    fx_est = (width / 2.0) / max(half_fov_h, 1e-6)

    if fov_vertical_deg:
        half_fov_v = np_radians(fov_vertical_deg) / 2.0
        fy_est = (height / 2.0) / max(half_fov_v, 1e-6)
        fy_source = Provenance.ESTIMATED
    else:
        fy_est = fx_est  # ASSUMED square pixels
        fy_source = Provenance.ASSUMED

    k_prov = Provenance(k_source)
    d_prov = Provenance(d_source)
    if fx is not None:
        fx_est = float(fx)
        k_prov = Provenance.CALIBRATED if k_source == "calibrated" else Provenance.ESTIMATED
    if fy is not None:
        fy_est = float(fy)
        fy_source = k_prov

    cx_v = float(cx) if cx is not None else width / 2.0
    cy_v = float(cy) if cy is not None else height / 2.0
    cx_prov = Provenance.CALIBRATED if cx is not None and k_source == "calibrated" else (
        Provenance.ESTIMATED if cx is not None else Provenance.ASSUMED
    )

    # fy_source is recorded only in comments; Intrinsics has a single K_source.
    _ = (fy_source, cx_prov)

    return Intrinsics(
        fx=float(fx_est),
        fy=float(fy_est),
        cx=cx_v,
        cy=cy_v,
        k1=float(k1),
        k2=float(k2),
        k3=float(k3),
        k4=float(k4),
        width=int(width),
        height=int(height),
        fov_horizontal_deg=float(fov_horizontal_deg),
        K_source=k_prov,
        D_source=d_prov,
    )


def np_radians(deg: float) -> float:
    import math

    return math.radians(deg)


def from_camera_yaml(cam: Dict[str, Any], frame_width: int, frame_height: int) -> Intrinsics:
    """Build Mode A/B intrinsics. Calibrated K/D in YAML win over FOV estimates."""
    img = cam.get("image") or {}
    width = int(frame_width or img.get("width") or 0)
    height = int(frame_height or img.get("height") or 0)
    fov = float(cam.get("fov_horizontal_deg") or 190.0)
    fov_v = cam.get("fov_vertical_deg")
    intra = cam.get("intrinsics") or {}
    dist = cam.get("distortion") or {}
    return estimate_intrinsics(
        width=width,
        height=height,
        fov_horizontal_deg=fov,
        fov_vertical_deg=float(fov_v) if fov_v else None,
        cx=intra.get("cx"),
        cy=intra.get("cy"),
        fx=intra.get("fx"),
        fy=intra.get("fy"),
        k1=float(dist.get("k1") or 0.0),
        k2=float(dist.get("k2") or 0.0),
        k3=float(dist.get("k3") or 0.0),
        k4=float(dist.get("k4") or 0.0),
        k_source=str(intra.get("source") or "estimated"),
        d_source=str(dist.get("source") or "estimated"),
    )


def describe_limitations() -> str:
    return (
        "Approximate intrinsics assume an equidistant (or lightly Kannala-Brandt) "
        "fisheye, a centered principal point, and a user-supplied FOV. They cannot "
        "recover lens decentering, per-element distortion, or the true focal length. "
        "Undistortion will look plausible near the image center and degrade toward "
        "the rim. Replace K and D with cv2.fisheye.calibrate() results for Mode B "
        "without changing the remap / projection API."
    )
