Four fisheye cameras → ~270° panoramic / see-through view (Java)
================================================================

This is an OpenCV **Java** prototype. It does not require a checkerboard,
vehicle measurements, or factory K/D. The primary output is a **cylindrical
~270° see-through strip**, not a bird's-eye view.

Build and run
-------------

```bash
mvn -q test
mvn -q exec:java -Dexec.args="synth"
mvn -q exec:java -Dexec.args="phase1"
mvn -q exec:java -Dexec.args="phase2"
mvn -q exec:java -Dexec.args="phase3"
mvn -q exec:java -Dexec.args="compare-projections"
mvn -q exec:java -Dexec.args="align"
mvn -q exec:java -Dexec.args="run --fov 270 --save-video output/pano.avi"
mvn -q exec:java -Dexec.args="serve 9090"
```

Then open http://localhost:9090/play (panorama) or http://localhost:9090/ (debug views + sliders).

Original calling convention is still accepted:

```bash
mvn -q exec:java -Dexec.args="9090 front.mov left.mov right.mov rear.mov"
```

Point `config/cameras.yaml` at real FRONT / LEFT / RIGHT / REAR videos when you have them.

Architecture
------------

```
INTRINSIC / FISHEYE CORRECTION  (Mode A estimated K,D  |  Mode B calibrated)
            ↓
      CYLINDRICAL PROJECTION     (precomputed mapX/mapY)
            ↓
       ALIGNMENT                 (offline ORB + RANSAC similarity)
            ↓
       STITCHING                 (feather blend, LEFT+FRONT+RIGHT for 270°)
```

Modules live under `src/main/java/fisheye270/` (`calibration`, `projection`,
`alignment`, `stitching`, `input`, `runtime`). See `docs/ARCHITECTURE.md`.

Mode A K/D (no target)
----------------------

```
fx = (width/2) / (fov_h_rad/2)     ESTIMATED
fy = fx                            ASSUMED square pixels
cx, cy = image center              ASSUMED
k1..k4 = 0                         ESTIMATED (equidistant)
```

Mode B: put calibrated fx/fy/cx/cy/k1..k4 in YAML with `source: calibrated`.
The remap API does not change.

Recommended projection: **cylindrical**. Perspective undistort is a debug view
only (cannot hold ~190° FOV). Homography is not the stitch model (parallax).
