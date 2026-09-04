from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fisheye_270.calibration.approximate_intrinsics import estimate_intrinsics
from fisheye_270.config_loader import load_all
from fisheye_270.input import FrameSynchronizer, SyntheticSource
from fisheye_270.runtime.processor import Processor
from fisheye_270.tools.synthetic import default_cameras, synthetic_frames
from fisheye_270.visualization.mosaic import mosaic_from_result, side_by_side


@pytest.fixture(scope="module")
def frames():
    seq, world = synthetic_frames(n=6, width=320, height=240)
    return seq, world


def test_intrinsics_from_fov():
    intr = estimate_intrinsics(640, 480, 180.0)
    assert intr.cx == 320
    assert abs(intr.fx - (320 / (np.pi / 2))) < 1e-4
    assert intr.K_source.value == "estimated"
    assert intr.D_source.value == "estimated"


def test_phase1_sync(frames):
    seq, _ = frames
    sources = {cid: SyntheticSource(cid, imgs) for cid, imgs in seq.items()}
    sync = FrameSynchronizer(sources)
    bundle = sync.next()
    assert set(bundle.frames) == {"FRONT", "LEFT", "RIGHT", "REAR"}
    assert bundle.missing == []
    for fr in bundle.frames.values():
        assert fr.ok and fr.image.shape[0] == 240


def test_phase2_front_undistort_changes_image(frames):
    seq, _ = frames
    cfg = load_all(ROOT / "config")
    proc = Processor(cfg)
    sources = {cid: SyntheticSource(cid, imgs) for cid, imgs in seq.items()}
    result = proc.process_bundle(FrameSynchronizer(sources).next())
    raw, corr = result.raw["FRONT"], result.undistorted["FRONT"]
    assert raw.shape == corr.shape
    assert np.mean(np.abs(raw.astype(int) - corr.astype(int))) > 1.0
    vis = side_by_side(raw, corr, "RAW", "CORRECTED")
    assert vis.shape[1] > raw.shape[1]


def test_phase3_all_cameras(frames):
    seq, _ = frames
    cfg = load_all(ROOT / "config")
    proc = Processor(cfg)
    sources = {cid: SyntheticSource(cid, imgs) for cid, imgs in seq.items()}
    result = proc.process_bundle(FrameSynchronizer(sources).next())
    assert set(result.undistorted) == {"FRONT", "LEFT", "RIGHT", "REAR"}


def test_cylindrical_panorama_not_bev(frames):
    seq, _ = frames
    cfg = load_all(ROOT / "config")
    cfg["cameras"]["output"]["width"] = 960
    cfg["cameras"]["output"]["height"] = 240
    cfg["cameras"]["output"]["fov_horizontal_deg"] = 270
    proc = Processor(cfg)
    sources = {cid: SyntheticSource(cid, imgs) for cid, imgs in seq.items()}
    result = proc.process_bundle(FrameSynchronizer(sources).next())
    assert result.panorama is not None
    h, w = result.panorama.shape[:2]
    assert w > h  # wide panorama, not a top-down square BEV
    assert proc.spec.model == "cylindrical"
    # usable pixels: not an empty canvas
    assert (result.panorama > 8).any()
    mosaic = mosaic_from_result(proc, result, "stitch")
    assert mosaic.size > 0


def test_missing_camera_does_not_crash(frames):
    seq, _ = frames
    cfg = load_all(ROOT / "config")
    proc = Processor(cfg)
    sources = {cid: SyntheticSource(cid, imgs) for cid, imgs in seq.items() if cid != "REAR"}
    result = proc.process_bundle(FrameSynchronizer(sources).next())
    assert "REAR" in result.missing
    assert result.panorama is not None


def test_yaml_mode_b_slots():
    cfg = load_all(ROOT / "config")
    front = cfg["cameras"]["cameras"]["FRONT"]
    assert "intrinsics" in front and "distortion" in front
    assert front["distortion"]["model"] == "fisheye"
    assert set(front["intrinsics"]).issuperset({"fx", "fy", "cx", "cy"})
