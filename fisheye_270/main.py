#!/usr/bin/env python3
"""CLI for the calibration-light 4-camera 270° panorama prototype.

Examples:
  python -m fisheye_270.main synth
  python -m fisheye_270.main phase1 --save output/phase1_raw.jpg
  python -m fisheye_270.main phase2 --save output/phase2_undistort.jpg
  python -m fisheye_270.main phase3 --save output/phase3_all.jpg
  python -m fisheye_270.main compare-projections --save output/phase4_proj.jpg
  python -m fisheye_270.main align
  python -m fisheye_270.main run --frames 40 --save-video output/pano.mp4
  python -m fisheye_270.main serve --port 9090
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import cv2

from fisheye_270.config_loader import load_all
from fisheye_270.input import FrameSynchronizer, SyntheticSource, open_configured_sources
from fisheye_270.runtime.processor import Processor
from fisheye_270.tools.synthetic import synthetic_frames, write_videos
from fisheye_270.visualization.mosaic import mosaic_from_result, side_by_side


def _ensure_synth(cfg, force: bool = False) -> None:
    cams = cfg["cameras"]["cameras"]
    missing = []
    for cid, cam in cams.items():
        path = Path((cam.get("source") or {}).get("path") or "")
        if not path.exists():
            missing.append(cid)
    if missing or force:
        print(f"Generating synthetic overlapping fisheye videos (missing={missing or 'forced'})...")
        frames, _world = synthetic_frames(n=48)
        paths = write_videos(Path("data/synthetic"), frames)
        for cid, path in paths.items():
            cams[cid]["source"] = {"type": "video", "path": str(path)}
            print(f"  {cid}: {path}")


def _sync_from_cfg(cfg, loop: bool = True) -> FrameSynchronizer:
    sources = open_configured_sources(cfg, loop=loop)
    opened = {k: v for k, v in sources.items() if v.is_open}
    if len(opened) < 1:
        frames, _ = synthetic_frames(n=24)
        opened = {cid: SyntheticSource(cid, seq) for cid, seq in frames.items()}
        print("Video files unreadable; using in-memory synthetic frames.")
    return FrameSynchronizer(opened)


def cmd_synth(cfg, args):
    _ensure_synth(cfg, force=True)
    return 0


def cmd_phase1(cfg, args):
    """Read four feeds and show them synchronously."""
    _ensure_synth(cfg)
    sync = _sync_from_cfg(cfg)
    proc = Processor(cfg)
    bundle = sync.next()
    result = proc.process_bundle(bundle)
    mosaic = mosaic_from_result(proc, result, "raw")
    _save(args.save, mosaic, default="output/phase1_raw.jpg")
    print("Phase 1: four synchronized raw views.")
    print(f"  cameras present: {list(result.raw.keys())}")
    print(f"  missing: {result.missing or 'none'}")
    print("  provenance: resolution=KNOWN (from frames); timestamps=ESTIMATED (frame index / fps)")
    sync.close()
    return 0


def cmd_phase2(cfg, args):
    """Approximate fisheye correction for FRONT only, RAW vs CORRECTED."""
    _ensure_synth(cfg)
    sync = _sync_from_cfg(cfg)
    proc = Processor(cfg)
    bundle = sync.next()
    result = proc.process_bundle(bundle)
    raw = result.raw["FRONT"]
    corr = result.undistorted["FRONT"]
    vis = side_by_side(raw, corr, "FRONT RAW", "FRONT PERSPECTIVE UNDISTORT")
    _save(args.save, vis, default="output/phase2_undistort.jpg")
    m = proc.models["FRONT"]
    print("Phase 2: FRONT approximate undistort (rectilinear debug view).")
    print(f"  K fx={m.intrinsics.fx:.1f} fy={m.intrinsics.fy:.1f} cx={m.intrinsics.cx:.1f} cy={m.intrinsics.cy:.1f}  [{m.intrinsics.K_source.value}]")
    print(f"  D k1..k4={[m.intrinsics.k1, m.intrinsics.k2, m.intrinsics.k3, m.intrinsics.k4]}  [{m.intrinsics.D_source.value}]")
    print("  FOV=ASSUMED from config; this view CANNOT hold a 190° FOV without crop/stretch.")
    print("  Stitching does NOT use this rectilinear image; it uses cylindrical maps.")
    sync.close()
    return 0


def cmd_phase3(cfg, args):
    _ensure_synth(cfg)
    sync = _sync_from_cfg(cfg)
    proc = Processor(cfg)
    bundle = sync.next()
    result = proc.process_bundle(bundle)
    vis = mosaic_from_result(proc, result, "undistorted")
    _save(args.save, vis, default="output/phase3_undistort_all.jpg")
    print("Phase 3: approximate undistort applied to FRONT, LEFT, RIGHT, REAR.")
    for cid, m in proc.models.items():
        print(f"  {cid}: {m.intrinsics.width}x{m.intrinsics.height} fx={m.intrinsics.fx:.1f} [{m.intrinsics.K_source.value}]")
    sync.close()
    return 0


def cmd_compare(cfg, args):
    _ensure_synth(cfg)
    sync = _sync_from_cfg(cfg)
    proc = Processor(cfg)
    bundle = sync.next()
    result = proc.process_bundle(bundle)
    vis = mosaic_from_result(proc, result, "compare_proj")
    _save(args.save, vis, default="output/phase4_projections.jpg")
    print("Phase 4: perspective vs cylindrical vs spherical on FRONT.")
    print("  RECOMMENDED for 270° driving view: cylindrical")
    print("  perspective: cannot represent >~120° without extreme rim stretch")
    print("  spherical/equirect: full 360, polar stretch, more cost")
    print("  cylindrical: natural 180–270° strip, vertical lines stay vertical")
    sync.close()
    return 0


def cmd_align(cfg, args):
    _ensure_synth(cfg)
    from fisheye_270.alignment.offline import run_offline_alignment
    sync = _sync_from_cfg(cfg, loop=False)
    report = run_offline_alignment(cfg, sync, save=not args.no_save)
    Path("output").mkdir(exist_ok=True)
    Path("output/alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Phase 5–8: pairwise ORB+RANSAC (+ photometric refine). Transforms frozen in cameras.yaml.")
    sync.close()
    return 0


def cmd_run(cfg, args):
    _ensure_synth(cfg)
    if args.fov:
        cfg["cameras"]["output"]["fov_horizontal_deg"] = float(args.fov)
    sync = _sync_from_cfg(cfg, loop=True)
    proc = Processor(cfg)
    writer = None
    n = int(args.frames)
    Path("output").mkdir(exist_ok=True)
    last = None
    for i in range(n):
        bundle = sync.next()
        if bundle is None:
            break
        result = proc.process_bundle(bundle)
        last = result
        if args.save_video and result.panorama is not None:
            if writer is None:
                h, w = result.panorama.shape[:2]
                writer = cv2.VideoWriter(
                    args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h)
                )
            writer.write(result.panorama)
        if i == 0 and result.panorama is not None:
            _save(args.save, mosaic_from_result(proc, result, "stitch"), default="output/phase12_pano.jpg")
            cv2.imwrite("output/phase8_seams.jpg", mosaic_from_result(proc, result, "seams"))
            cv2.imwrite("output/phase5_overlap.jpg", mosaic_from_result(proc, result, "overlap"))
    if writer is not None:
        writer.release()
        print(f"Wrote {args.save_video}")
    print("Phase 9–13: masks, photometric gain/offset, feather blend, runtime LUTs.")
    print("  stats:", proc.perf.snapshot())
    if last:
        print("  missing cameras:", last.missing or "none")
        print("  timings_ms:", {k: round(v, 2) for k, v in last.timings_ms.items()})
    sync.close()
    return 0


def cmd_serve(cfg, args):
    _ensure_synth(cfg)
    from fisheye_270.visualization.web_viewer import create_app
    sync = _sync_from_cfg(cfg, loop=True)
    proc = Processor(cfg)
    app = create_app(proc, sync.next)
    print(f"Debug UI -> http://127.0.0.1:{args.port}/")
    app.run(host="0.0.0.0", port=int(args.port), threaded=True, debug=False)
    return 0


def _save(path: str | None, image, default: str) -> None:
    out = Path(path or default)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)
    print(f"Wrote {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="4-camera fisheye → ~270° panorama (calibration-light)")
    p.add_argument("--config-dir", default="config")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth")
    for name in ("phase1", "phase2", "phase3", "compare-projections"):
        sp = sub.add_parser(name)
        sp.add_argument("--save", default=None)
    al = sub.add_parser("align")
    al.add_argument("--no-save", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--frames", default=24)
    run.add_argument("--save", default=None)
    run.add_argument("--save-video", default=None)
    run.add_argument("--fov", default=None, help="output horizontal FOV degrees")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", default=9090)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_all(Path(args.config_dir))
    cmds = {
        "synth": cmd_synth,
        "phase1": cmd_phase1,
        "phase2": cmd_phase2,
        "phase3": cmd_phase3,
        "compare-projections": cmd_compare,
        "align": cmd_align,
        "run": cmd_run,
        "serve": cmd_serve,
    }
    return cmds[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
