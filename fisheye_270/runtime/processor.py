from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..calibration.approximate_intrinsics import from_camera_yaml
from ..calibration.camera_model import Alignment2D, CameraModel, Pose, Provenance
from ..calibration.fisheye_model import remap, undistort_maps
from ..config_loader import camera_cfg, camera_ids
from ..projection.projection_maps import MapCache, PanoramaSpec
from ..stitching.blend import (
    CAMERA_COLORS,
    apply_gain_offset,
    distance_feather_mask,
    estimate_gain_offset,
    feather_composite,
    hard_mask,
    seam_visualization,
)
from .performance import PerformanceMonitor


@dataclass
class PipelineResult:
    bundle_index: int
    raw: Dict[str, np.ndarray]
    undistorted: Dict[str, np.ndarray]
    projected: Dict[str, np.ndarray]
    valid: Dict[str, np.ndarray]
    panorama: Optional[np.ndarray]
    seams: Optional[np.ndarray]
    missing: List[str] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)


class Processor:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.cache = MapCache()
        self.undistort_cache: Dict[str, Tuple] = {}
        self.models: Dict[str, CameraModel] = {}
        self.photometric: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._weight_cache: Dict[str, np.ndarray] = {}
        self._seams: Optional[np.ndarray] = None
        self.perf = PerformanceMonitor()
        out = cfg["cameras"].get("output") or {}
        proj = cfg["projection"]
        self.spec = PanoramaSpec(
            width=int(out.get("width") or 1920),
            height=int(out.get("height") or 540),
            fov_h_deg=float(out.get("fov_horizontal_deg") or 270),
            fov_v_deg=float(out.get("fov_vertical_deg") or 70),
            yaw_center_deg=float(out.get("yaw_center_deg") or 0),
            model=str(proj.get("model") or "cylindrical"),
        )
        self.blend_width = int((cfg["stitching"].get("seam") or {}).get("blend_width_px") or 80)
        self.seam_mode = str((cfg["stitching"].get("seam") or {}).get("mode") or "feather")

    def _model_for(self, camera_id: str, frame: np.ndarray) -> CameraModel:
        cam = camera_cfg(self.cfg, camera_id)
        h, w = frame.shape[:2]
        existing = self.models.get(camera_id)
        if existing and existing.intrinsics.width == w and existing.intrinsics.height == h:
            return existing
        intr = from_camera_yaml(cam, w, h)
        pose = cam.get("pose") or {}
        al = cam.get("alignment") or {}
        model = CameraModel(
            camera_id=camera_id,
            intrinsics=intr,
            pose=Pose(
                yaw_deg=float(pose.get("yaw_deg") or 0.0),
                pitch_deg=float(pose.get("pitch_deg") or 0.0),
                roll_deg=float(pose.get("roll_deg") or 0.0),
                tx=float(pose.get("tx") or 0.0),
                ty=float(pose.get("ty") or 0.0),
                tz=float(pose.get("tz") or 0.0),
                source=Provenance(str(pose.get("source") or "assumed")),
            ),
            alignment=Alignment2D(
                dx=float(al.get("dx") or 0.0),
                dy=float(al.get("dy") or 0.0),
                rotation_deg=float(al.get("rotation_deg") or 0.0),
                scale=float(al.get("scale") or 1.0),
                source=Provenance(str(al.get("source") or "estimated")),
            ),
            enabled=bool(cam.get("enabled", True)),
        )
        self.models[camera_id] = model
        return model

    def undistort(self, camera_id: str, frame: np.ndarray) -> np.ndarray:
        model = self._model_for(camera_id, frame)
        key = (camera_id, model.intrinsics.width, model.intrinsics.height,
               model.intrinsics.fx, model.intrinsics.k1)
        cached = self.undistort_cache.get(camera_id)
        if cached is None or cached[0] != key:
            map1, map2, _ = undistort_maps(model.intrinsics)
            self.undistort_cache[camera_id] = (key, map1, map2)
        _, map1, map2 = self.undistort_cache[camera_id]
        return remap(frame, map1, map2)

    def project(self, camera_id: str, frame: np.ndarray, model_name: Optional[str] = None):
        cam = self._model_for(camera_id, frame)
        spec = self.spec
        if model_name and model_name != spec.model:
            # comparison path — do not pollute the primary cache key unnecessarily
            from ..projection.projection_maps import fisheye_to_panorama_maps
            tmp = PanoramaSpec(
                width=spec.width, height=spec.height, fov_h_deg=spec.fov_h_deg,
                fov_v_deg=spec.fov_v_deg, yaw_center_deg=spec.yaw_center_deg, model=model_name,
            )
            mx, my, valid = fisheye_to_panorama_maps(cam, tmp, model_name)
        else:
            mx, my, valid = self.cache.get(cam, spec, spec.model)
        return remap(frame, mx, my), valid

    def process_bundle(self, bundle) -> PipelineResult:
        t0 = perf_counter()
        raw: Dict[str, np.ndarray] = {}
        undistorted: Dict[str, np.ndarray] = {}
        projected: Dict[str, np.ndarray] = {}
        valid: Dict[str, np.ndarray] = {}
        t_und = t_proj = 0.0

        for cid in camera_ids(self.cfg):
            img = bundle.image(cid)
            if img is None:
                continue
            raw[cid] = img
            t1 = perf_counter()
            undistorted[cid] = self.undistort(cid, img)
            t_und += perf_counter() - t1
            t2 = perf_counter()
            projected[cid], valid[cid] = self.project(cid, img)
            t_proj += perf_counter() - t2

        t3 = perf_counter()
        panorama, seams = self._stitch(projected, valid)
        t_st = perf_counter() - t3

        timings = {
            "undistort_ms": t_und * 1000,
            "project_ms": t_proj * 1000,
            "stitch_ms": t_st * 1000,
            "total_ms": (perf_counter() - t0) * 1000,
        }
        self.perf.record(timings["total_ms"])
        return PipelineResult(
            bundle_index=bundle.index,
            raw=raw,
            undistorted=undistorted,
            projected=projected,
            valid=valid,
            panorama=panorama,
            seams=seams,
            missing=list(bundle.missing),
            timings_ms=timings,
        )

    def _stitch(
        self, projected: Dict[str, np.ndarray], valid: Dict[str, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not projected:
            return None, None
        photo_cfg = self.cfg["stitching"].get("photometric") or {}
        ref_id = str(photo_cfg.get("reference") or "FRONT")
        layers = []
        weights = []
        weight_map = {}
        # 270° front-centered see-through: LEFT+FRONT+RIGHT.
        # REAR still sees the canvas edges (190° lens) and would double-image the sides.
        order = [c for c in ("LEFT", "FRONT", "RIGHT") if c in projected]
        if self.spec.fov_h_deg > 270.5 and "REAR" in projected:
            order.append("REAR")

        ref_img = projected.get(ref_id)
        ref_valid = valid.get(ref_id)

        for cid in order:
            img = projected[cid]
            val = valid[cid]
            if photo_cfg.get("enabled", True) and ref_img is not None and cid != ref_id:
                if cid in self.photometric:
                    g, o = self.photometric[cid]
                else:
                    overlap = ((val > 0) & (ref_valid > 0)).astype(np.uint8) if ref_valid is not None else val
                    g, o = estimate_gain_offset(
                        img, ref_img, overlap,
                        max_gain=float(photo_cfg.get("max_gain") or 2.5),
                        max_offset=float(photo_cfg.get("max_offset") or 40.0),
                    )
                    self.photometric[cid] = (g, o)
                img = apply_gain_offset(img, g, o)
            cache_key = (self.blend_width, self.seam_mode, int(val.sum()))
            wt = self._weight_cache.get(cid)
            stored_key = self._weight_cache.get(f"{cid}_key")
            if wt is None or stored_key != cache_key:
                wt = hard_mask(val) if self.seam_mode == "hard" else distance_feather_mask(val, self.blend_width)
                self._weight_cache[cid] = wt
                self._weight_cache[f"{cid}_key"] = cache_key
            layers.append(img)
            weights.append(wt)
            weight_map[cid] = wt

        pano = feather_composite(layers, weights)
        if self._seams is None or set(weight_map) != getattr(self, "_seam_ids", set()):
            self._seams = seam_visualization(weight_map, CAMERA_COLORS)
            self._seam_ids = set(weight_map)
        return pano, self._seams

    def label_quad(self, images: Dict[str, np.ndarray], titles: Optional[Dict[str, str]] = None) -> np.ndarray:
        tiles = []
        for cid in ("FRONT", "LEFT", "RIGHT", "REAR"):
            img = images.get(cid)
            if img is None:
                img = np.zeros((240, 320, 3), np.uint8)
            else:
                img = img.copy()
            h, w = img.shape[:2]
            scale = 320 / max(w, 1)
            img = cv2.resize(img, (320, max(int(h * scale), 1)))
            title = (titles or {}).get(cid, cid)
            cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(img)
        h = max(t.shape[0] for t in tiles)
        w = max(t.shape[1] for t in tiles)
        norm = []
        for t in tiles:
            canvas = np.zeros((h, w, 3), np.uint8)
            canvas[: t.shape[0], : t.shape[1]] = t
            norm.append(canvas)
        top = np.hstack(norm[0:2])
        bot = np.hstack(norm[2:4])
        return np.vstack([top, bot])
