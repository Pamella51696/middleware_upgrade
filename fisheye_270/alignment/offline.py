from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..alignment.align import align_pair, draw_matches
from ..config_loader import camera_cfg, camera_ids, save_cameras
from ..input import FrameSynchronizer
from ..runtime.processor import Processor


def run_offline_alignment(cfg: Dict[str, Any], sync: FrameSynchronizer, save: bool = True) -> Dict[str, Any]:
    """Estimate 2D similarity per camera vs FRONT on a single frozen bundle."""
    bundle = sync.next()
    if bundle is None:
        raise RuntimeError("no frames available for alignment")
    proc = Processor(cfg)
    result = proc.process_bundle(bundle)
    stitch_cfg = cfg["stitching"]
    pairs: List = list((stitch_cfg.get("pairs") or [["LEFT", "FRONT"], ["FRONT", "RIGHT"]]))
    report: Dict[str, Any] = {"pairs": [], "method_notes": []}
    vis_dir = Path("output/alignment")
    vis_dir.mkdir(parents=True, exist_ok=True)

    for a, b in pairs:
        ia, ib = result.projected.get(a), result.projected.get(b)
        va, vb = result.valid.get(a), result.valid.get(b)
        if ia is None or ib is None:
            report["method_notes"].append(f"{a}-{b}: skipped (missing camera)")
            continue
        overlap = ((va > 0) & (vb > 0)).astype(np.uint8)
        dbg = align_pair(ia, ib, overlap, stitch_cfg, pair=(a, b))
        vis = draw_matches(ia, ib, dbg)
        cv2.imwrite(str(vis_dir / f"matches_{a}_{b}.jpg"), vis)
        report["pairs"].append({
            "pair": [a, b],
            "method": dbg.method,
            "inliers": dbg.transform.inliers,
            "confidence": dbg.transform.confidence,
            "dx": dbg.transform.dx,
            "dy": dbg.transform.dy,
            "rotation_deg": dbg.transform.rotation_deg,
            "scale": dbg.transform.scale,
            "overlap_l1": dbg.overlap_error,
        })
        # Apply to the non-FRONT camera when possible
        target = a if b == "FRONT" else (b if a == "FRONT" else a)
        sign = 1.0 if target == a else -1.0
        if dbg.transform.confidence > 0 or dbg.method.startswith("phase"):
            node = camera_cfg(cfg, target).setdefault("alignment", {})
            node["dx"] = float(sign * dbg.transform.dx)
            node["dy"] = float(sign * dbg.transform.dy)
            node["rotation_deg"] = float(sign * dbg.transform.rotation_deg)
            node["scale"] = float(dbg.transform.scale)
            node["source"] = "estimated"
            report["method_notes"].append(f"{target}: wrote alignment ({dbg.method})")
        else:
            report["method_notes"].append(f"{a}-{b}: low confidence, kept previous transform")

    if save:
        out = Path(cfg["config_dir"]) / "cameras.yaml"
        save_cameras(cfg, out)
        report["saved"] = str(out)
    return report
