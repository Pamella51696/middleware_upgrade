# Jetson Orin Nano path (Phase 14 — not implemented in this Python lab)

Do **not** introduce TensorRT. There is no neural network in fisheye correction or stitching.

Recommended stack:

```
4 × CSI/USB
    → GStreamer (nvarguscamerasrc or v4l2src) + NTP/hardware timestamps
    → GPU upload (already NVMM if using nvvidconv)
    → OpenCV CUDA remap × 4   (precomputed mapX/mapY as GpuMat)
    → CUDA multiply-add blend (or cv::cuda::blend)
    → NVENC / display
```

DeepStream: only if the rest of the vehicle computer already uses it for detectors. It does not buy anything for LUT stitching.

OpenCV CUDA vs custom kernels: start with `cv::cuda::remap`. Write a fused kernel only if Nsight says you are memory-bound on four separate remaps.

Python prototype maps are already the exact buffers a C++ port should consume (same YAML → same map builder).
