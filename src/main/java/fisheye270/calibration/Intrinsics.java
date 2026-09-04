package fisheye270.calibration;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * OpenCV camera matrix K and Kannala–Brandt fisheye D.
 * Mode A fills these from FOV + resolution. Mode B overwrites from
 * Calib3d.fisheye_calibrate without changing remap callers.
 */
public final class Intrinsics {
    public double fx, fy, cx, cy;
    public double k1, k2, k3, k4;
    public int width, height;
    public double fovHorizontalDeg;
    public Provenance kSource = Provenance.ESTIMATED;
    public Provenance dSource = Provenance.ESTIMATED;

    public Mat K() {
        Mat k = Mat.eye(3, 3, CvType.CV_64FC1);
        k.put(0, 0, fx);
        k.put(1, 1, fy);
        k.put(0, 2, cx);
        k.put(1, 2, cy);
        return k;
    }

    public Mat D() {
        Mat d = new Mat(4, 1, CvType.CV_64FC1);
        d.put(0, 0, k1);
        d.put(1, 0, k2);
        d.put(2, 0, k3);
        d.put(3, 0, k4);
        return d;
    }
}
