Four fisheye cameras → ~270° panoramic / see-through view
=========================================================

This repository now contains two layers:

1. `VideoStreamingServer.java` — the existing Java MJPEG stitcher (unchanged).
2. `fisheye_270/` — a **calibration-light Python prototype** that separates
   intrinsics, panoramic projection, alignment, and stitching.

The Python system is the architecture requested for the automotive prototype.
It does **not** require a checkerboard, ChArUco board, vehicle measurements, or
factory K/D files. It also does **not** produce a bird's-eye view as the
primary output.

Quick start
-----------

```bash
pip install -r requirements.txt
python -m fisheye_270.main synth
python -m fisheye_270.main phase1 --save output/phase1_raw.jpg
python -m fisheye_270.main phase2 --save output/phase2_undistort.jpg
python -m fisheye_270.main phase3 --save output/phase3_undistort_all.jpg
python -m fisheye_270.main compare-projections --save output/phase4_projections.jpg
python -m fisheye_270.main align
python -m fisheye_270.main run --frames 40 --save-video output/pano.mp4 --fov 270
python -m fisheye_270.main serve --port 9090   # debug UI
python -m pytest tests -q
```

Point `config/cameras.yaml` `source.path` at your real `FRONT` / `LEFT` /
`RIGHT` / `REAR` videos when you have them. Until then the CLI generates
overlapping synthetic fisheye feeds in `data/synthetic/`.

Point-by-point design answers (section 33)
------------------------------------------

### 1–2. What intrinsic calibration is, and whether you need it now

Intrinsic calibration estimates the camera matrix **K** (focal length, principal
point) and distortion **D** so that a pixel can be turned into a ray in camera
space. Formal calibration (`cv2.fisheye.calibrate`) **is not strictly required**
for a visually plausible prototype. It **is** required for metric geometry,
stable undistortion at the rim, and any later fusion with ranging sensors.

### 3. If K and D are unavailable

The pipeline still runs (Mode A). It **assumes** an equidistant fisheye, a
centered principal point, and a user FOV (default 190°). Undistortion and
panoramic projection will look reasonable in the middle of the image and will
be wrong near the lens ring, on decentering, and wherever the real FOV differs.

### 4–6. Approximate parameters now, real K/D later

Initialize:

```
fx = (width/2) / (fov_h_rad/2)     # ESTIMATED  (equidistant)
fy = fx                            # ASSUMED square pixels
cx, cy = width/2, height/2         # ASSUMED
k1..k4 = 0                         # ESTIMATED (pure equidistant)
```

Tune FOV, fx/fy, cx/cy, k1..k4 in the debug UI. Mode B: paste calibrated K/D
into `config/cameras.yaml` (`intrinsics.source: calibrated`). The remap API
does not change.

### Recommended projection

**Cylindrical** (Approach B), not BEV and not rectilinear.

| | Distortion | FOV | Cost | Overlap | Driving view |
|---|---|---|---|---|---|
| Perspective undistort | Rim → infinity | Cannot hold ~190° | Low | Poor | Debug only |
| **Cylindrical** | Vertical lines OK | 180–270° natural | LUT remap | Horizontal strips | **Best** |
| Spherical / equirect | Poles stretch | 360° | LUT, more cache | Good | Use if you need full 360° |

### What can be estimated without the vehicle

KNOWN: resolution, pixel format, (approx) fps.
ASSUMED: 190° FOV, centered cx/cy, zero baseline, yaw 0/−90/+90/+180.
ESTIMATED: fx/fy from FOV, overlap 2D similarity, photometric gain/offset, seams.

### What cannot be solved reliably without calibration / the car

True K/D, baseline (tx,ty,tz), ground-plane metric, parallax-free seams on
nearby objects, centimeter alignment, OEM surround-view accuracy.

### Automatic alignment

Offline only: project → ORB + BFMatcher + RANSAC similarity → photometric
refine in overlap → reject physically huge transforms → write YAML.
Runtime uses **frozen** maps. Low texture → keep last transform (or identity).

Homography is **not** the primary model: the scene is not planar, cameras
have parallax, and fisheye rays are not perspective.

See `docs/ARCHITECTURE.md` for the full math, Jetson plan, and failure handling.
