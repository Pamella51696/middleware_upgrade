Four fisheye cameras → ~270° panoramic / see-through view (Java)
================================================================

This is an OpenCV **Java** prototype. It does not require a checkerboard,
vehicle measurements, or factory K/D. The primary output is a **cylindrical
~270° see-through strip**, not a bird's-eye view.

Build and run (Windows, official OpenCV JAR only)
------------------------------------------------

Do **not** compile only `*.java` inside `src\main\java\fisheye270`. That folder
is just the top-level package; the rest of the code is in subfolders, and there
is **no SnakeYAML JAR**. Config is parsed in-process.

From the **repo root** (the folder that contains `config\` and `pom.xml`):

```bat
set OPENCV_JAR=C:\Users\ps95973\Downloads\opencv\build\java\opencv-490.jar
set OPENCV_NATIVE=C:\Users\ps95973\Downloads\opencv\build\java\x64
compile.bat

java -cp "%OPENCV_JAR%;out" -Djava.library.path="%OPENCV_NATIVE%" fisheye270.VideoStreamingServer 9090 front.mov left.mov right.mov rear.mov
```

Or in PowerShell: `.\compile.ps1`

Equivalent one-liner:

```bat
dir /s /b src\main\java\*.java > out\sources.txt
javac -encoding UTF-8 -cp "%OPENCV_JAR%" -d out @out\sources.txt
```

`javac -cp opencv-490.jar *.java` from `fisheye270\` will fail: missing packages
and (previously) missing `Yaml`.

Maven (optional)
----------------

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
