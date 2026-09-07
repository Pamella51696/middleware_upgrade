package fisheye270.calibration;

/**
 * Mode A: equidistant fisheye from image size and assumed FOV.
 *   r_px = f * theta
 *   (width/2) = fx * (fov_h/2)
 * Principal point ASSUMED at the frame center. Not a substitute for
 * Calib3d.fisheye_calibrate.
 */
public final class ApproximateIntrinsics {
    private ApproximateIntrinsics() {}

    public static Intrinsics fromSizeAndFov(
            int width, int height, double fovHDeg,
            Double fx, Double fy, Double cx, Double cy,
            double k1, double k2, double k3, double k4,
            Provenance kSource, Provenance dSource) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("width/height must be KNOWN from the first frame");
        }
        Intrinsics i = new Intrinsics();
        i.width = width;
        i.height = height;
        i.fovHorizontalDeg = fovHDeg;
        double half = Math.toRadians(fovHDeg) / 2.0;
        i.fx = fx != null ? fx : (width / 2.0) / Math.max(half, 1e-6);
        i.fy = fy != null ? fy : i.fx;
        i.cx = cx != null ? cx : width / 2.0;
        i.cy = cy != null ? cy : height / 2.0;
        i.k1 = k1;
        i.k2 = k2;
        i.k3 = k3;
        i.k4 = k4;
        i.kSource = kSource;
        i.dSource = dSource;
        return i;
    }
}
