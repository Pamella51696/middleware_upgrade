# Architecture — 4-camera fisheye → ~270° panorama

Status of every geometric quantity is one of: **KNOWN / ASSUMED / ESTIMATED / CALIBRATED / UNKNOWN**.

This prototype is **visually plausible, reasonably aligned, temporally stable**. It is not OEM surround-view.

---

## A. Block diagram

```
FRONT LEFT RIGHT REAR
        │
        ▼
 Camera Input Manager  (video | image sequence | live later)
        ▼
 Frame timestamp (file: index/fps ESTIMATED; live: capture PTS KNOWN)
        ▼
 Frame synchronizer (lock-step or nearest PTS; hold-last on drop)
        ▼
 ┌──────────────────────────────────────────────────────────┐
 │  PER CAMERA (independent, swappable)                     │
 │    1. INTRINSIC / FISHEYE   K, D  → optional perspective │
 │    2. PROJECTION            fisheye rays → cylinder LUT  │
 │    3. ALIGNMENT             frozen 2D similarity on pano │
 │    4. MASK                  validity + feather weights   │
 └──────────────────────────────────────────────────────────┘
        ▼
 Photometric gain/offset from overlap vs FRONT
        ▼
 Feather (or hard) composite
        ▼
 ~270° cylindrical panorama  (FOV configurable 180/220/240/270/360)
```

Bird's-eye view is **not** in this path. A BEV would require a ground plane and camera height (UNKNOWN).

---

## B. Frame-by-frame data flow (runtime)

1. Read one frame per live camera. A dead camera is omitted; last good frame may be held ≤ `hold_last_frame_ms`.
2. Look up precomputed `mapX, mapY` (and a validity mask) for that camera.
3. `cv2.remap` fisheye → panorama canvas.
4. Apply cached photometric `gain, offset` (optionally slowly updated).
5. Weighted blend. No ORB, no RANSAC, no map rebuild.

Offline (once per install / when you retune): feature match → RANSAC → photometric refine → write YAML → rebuild LUTs.

---

## C. Mathematical model

### C.1 Intrinsics (OpenCV fisheye / Kannala–Brandt)

```
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
D = [k1, k2, k3, k4]
```

A camera ray `(x, y, z)` (not necessarily z>0 — FOV may exceed 180°):

```
r     = hypot(x, y)
theta = atan2(r, z)                          # angle from optical axis
theta_d = theta * (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
u = fx * theta_d * x / r + cx
v = fy * theta_d * y / r + cy
```

`cv2.fisheye.calibrate` estimates K,D from a target (Mode B).
`estimateNewCameraMatrixForUndistortRectify` + `initUndistortRectifyMap` + `remap` produce a **rectilinear** image. That image is a **debug view**. It cannot represent a ~190° lens without cropping FOV (`balance≈0`) or sending the rim toward infinity (`balance≈1`).

**Mode A** (no target):

```
fx = (width/2) / (fov_h / 2)     ESTIMATED, equidistant
fy = fx                          ASSUMED square pixels
cx, cy = image center            ASSUMED
k1..k4 = 0                       ESTIMATED (or small k1 from the UI)
```

Limitations: decentering, real FOV, non-equidistant glass, and the true focal length are not recovered. Rim geometry will be wrong. Alignment can hide some of that in overlap, not at the optical axis extremes.

### C.2 Panoramic coordinates (not BEV)

Canvas pixel `(x, y)`:

```
theta = yaw_center - fov_h/2 + x * fov_h / width     # heading
phi   = (y/height - 1/2) * fov_v                     # pitch (Y down)
```

Vehicle ray (X right, Y down, Z forward):

* cylindrical: direction `(sin θ, tan φ, cos θ)` then normalized
* spherical: `(sin θ cos φ, sin φ, cos θ cos φ)`

Then `ray_cam = R_cam_to_vehicleᵀ · ray_vehicle`, then fisheye project. The LUT stores that `(u,v)` per pano pixel.

Default yaws **ASSUMED**: FRONT 0°, RIGHT +90°, LEFT −90°, REAR 180°.
Default output: `yaw_center=0`, `fov_h=270°` → about −135°…+135° (LEFT+FRONT+RIGHT). REAR is still processed for debug, 360° mode, and LEFT–REAR / RIGHT–REAR alignment.

### C.3 Alignment

After projection, neighbors overlap in **image** space. We estimate a **similarity** (dx, dy, rotation, scale), not a homography.

A homography assumes a plane (or a pure rotation about the same optical center). Here:

* cameras have a baseline (**UNKNOWN** → we set translation = 0, zero-parallax);
* the world is not planar (road + cars + buildings);
* residual fisheye error is not a homography.

Homography overfits asphalt and breaks as soon as something is at a different depth (parallax). Similarity is weaker and more stable. Affine is the next step if needed.

RANSAC inliers → confidence. If `inliers < min` or the transform exceeds `max_abs_dx/dy/rotation/scale`, we fall back to phase correlation (translation) or **hold the last stored transform**. Runtime does not randomly re-align on snow/night/water.

Photometric refine: coordinate descent on overlap L1, small steps, stop when steps collapse or error stalls.

### C.4 Photometry and blending

```
I' = gain ⊙ I + offset
```

`gain, offset` from overlap mean/std vs FRONT (clipped).

Seams: **feather** using a distance transform of the validity mask (default). Hard seam is available. Multiband is deferred until feather is visibly insufficient — it costs more bands and can ghost on parallax.

---

## D. Software modules

| Module | Responsibility |
|---|---|
| `input/` | Video / images / synthetic; sync; missing-camera policy |
| `calibration/` | `CameraModel`, Mode A K/D, OpenCV fisheye LUTs |
| `projection/` | Cylinder / sphere maps, cache keyed on K,D,pose,alignment |
| `alignment/` | ORB, BFMatcher, RANSAC similarity, refine, offline YAML |
| `stitching/` | Masks, gain/offset, feather, seam viz |
| `runtime/` | Processor, metrics, performance |
| `visualization/` | Eight debug views + Flask sliders |
| `tools/synthetic.py` | Stand-in overlapping fisheye world |
| `config/*.yaml` | All tunables, Mode B slots (`fx: calibrated`) |

No God class: `Processor` only sequences the four stages.

---

## E. Configuration

See `config/cameras.yaml`, `projection.yaml`, `stitching.yaml`.
Every intrinsic may be `null` (Mode A fill) or a number with `source: calibrated` (Mode B).

---

## F–I. Runnable pieces

| Command | Phase |
|---|---|
| `python -m fisheye_270.main phase1` | 1 sync raw |
| `… phase2` | 2 FRONT undistort |
| `… phase3` | 3 all four undistort |
| `… compare-projections` | 4 perspective/cyl/sphere |
| `… align` | 5–8 overlap, ORB, RANSAC, freeze YAML |
| `… run --fov 270` | 9–13 masks, photo, blend, pano video |
| `… serve` | debug UI views 1–8 + sliders |

---

## J. Performance → C++ / CUDA / Jetson

Python is the algorithm lab. The hot path is already LUT `remap` (good). Next:

1. Keep maps as `CV_16SC2` + `INTER_LINEAR` (faster gather).
2. `cv2.cuda.remap` (OpenCV CUDA) for four remaps + blend on Orin Nano. No TensorRT — there is no network.
3. Optional: one fused CUDA kernel (remap + weight + accumulate) to cut memory traffic.
4. Capture: **GStreamer** (`nvarguscamerasrc` / V4L2) into CUDA buffers. **DeepStream is not required** unless you already use it for other analytics.
5. C++ port of `Processor` + same YAML. Python UI can stay as a tuner that writes YAML.

Expected bottleneck: four 1080p remaps + a 1920×540 blend. Orin Nano GPU remap should be real-time; CPU Python is the prototype (tens of ms/frame at 640×480 synthetic).

Profile fields: FPS, latency_ms, CPU%, RSS, GPU% (GPU via `tegrastats` on device; `-1` here).

---

## Failure handling

| Event | Behavior |
|---|---|
| Camera disconnect / unreadable file | Mark missing; stitch remaining; no crash |
| Dropped / corrupt frame | Hold last; skip if none |
| Feature match fail / low texture | Keep stored transform |
| Exposure pop | Photometric clip + optional slow IIR later |
| Partial obstruction | Validity mask already drops unmapped pixels |
| FPS mismatch | Lock-step files; nearest PTS live |

---

## Alignment method comparison (why ORB + similarity)

| Method | Pros | Cons | Role |
|---|---|---|---|
| ORB + BF + RANSAC | Fast, no SIFT license, OK on structure | Fails on asphalt/night | Default offline |
| AKAZE | More stable binary desc. | Slower | Alternative detector flag |
| SIFT | Stronger | License/perf | Optional later |
| Optical flow | Dense | Needs texture + motion; not for freeze | Not runtime |
| Phase correlation | Works on weak texture (translation) | No rotation/scale | Fallback |
| Template match | Simple | Scale/rot fragile | Not used |
| Homography | Fits planes well | Parallax disaster | Explicitly avoided |
| Similarity / affine | Matches cylindrical residual | Cannot fix large parallax | **Chosen** |

---

## Parameter provenance cheat sheet

| Quantity | Status |
|---|---|
| Frame width/height | KNOWN |
| Camera FOV | UNKNOWN → ASSUMED 190° |
| fx, fy | ESTIMATED from FOV (or CALIBRATED later) |
| cx, cy | ASSUMED center (or CALIBRATED) |
| k1..k4 | ESTIMATED 0 (or CALIBRATED) |
| Camera (tx,ty,tz) | UNKNOWN → 0 (zero parallax) |
| Yaw 0/±90/180 | ASSUMED |
| Alignment dx,dy,… | ESTIMATED offline |
| Output 270° | ASSUMED product target |
