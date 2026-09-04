# Jetson Orin Nano path

The runtime is Java + OpenCV remap LUTs. Do **not** introduce TensorRT.

Recommended device stack:

```
4 × CSI/USB
    → GStreamer (nvarguscamerasrc or v4l2src)
    → GPU upload
    → OpenCV CUDA remap × 4  (precomputed mapX/mapY)
    → CUDA multiply-add blend
    → NVENC / display
```

DeepStream is optional and only useful if the rest of the vehicle computer already uses it.

The Java `ProjectionMaps` buffers are the same maps a CUDA port should consume (same YAML → same geometry).
