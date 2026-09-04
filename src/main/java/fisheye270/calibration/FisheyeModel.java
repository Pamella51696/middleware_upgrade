package fisheye270.calibration;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * OpenCV fisheye LUTs (perspective debug path) and Kannala–Brandt projection
 * that supports FOV &gt; 180° (z can be negative).
 *
 * Runtime stitching must NOT use the perspective maps — use cylindrical maps.
 */
public final class FisheyeModel {
    private FisheyeModel() {}

    public static Mat[] undistortMaps(Intrinsics intr, double balance, double fovScale) {
        Mat K = intr.K();
        Mat D = intr.D();
        Mat R = Mat.eye(3, 3, CvType.CV_64FC1);
        Mat P = new Mat();
        Size src = new Size(intr.width, intr.height);
        Calib3d.fisheye_estimateNewCameraMatrixForUndistortRectify(
                K, D, src, R, P, balance, src, fovScale);
        Mat map1 = new Mat();
        Mat map2 = new Mat();
        Calib3d.fisheye_initUndistortRectifyMap(
                K, D, R, P, src, CvType.CV_32FC1, map1, map2);
        K.release();
        D.release();
        R.release();
        P.release();
        return new Mat[]{map1, map2};
    }

    public static Mat remap(Mat src, Mat mapX, Mat mapY) {
        Mat dst = new Mat();
        Imgproc.remap(src, dst, mapX, mapY, Imgproc.INTER_LINEAR);
        return dst;
    }

    /**
     * Project a camera-frame ray to a fisheye pixel.
     * @return false if outside the image or beyond the assumed FOV
     */
    public static boolean project(Intrinsics intr, double x, double y, double z, float[] uv) {
        double r = Math.hypot(x, y);
        double theta = Math.atan2(r, z);
        double t2 = theta * theta;
        double t4 = t2 * t2;
        double scale = 1.0 + intr.k1 * t2 + intr.k2 * t4 + intr.k3 * t4 * t2 + intr.k4 * t4 * t4;
        double thetaD = theta * scale;
        double u;
        double v;
        if (r <= 1e-12) {
            u = intr.cx;
            v = intr.cy;
        } else {
            u = intr.fx * thetaD * x / r + intr.cx;
            v = intr.fy * thetaD * y / r + intr.cy;
        }
        double maxTheta = Math.toRadians(Math.min(Math.max(intr.fovHorizontalDeg * 0.55, 80.0), 170.0));
        if (!Double.isFinite(u) || !Double.isFinite(v) || theta >= maxTheta
                || u < 0 || v < 0 || u >= intr.width || v >= intr.height) {
            return false;
        }
        uv[0] = (float) u;
        uv[1] = (float) v;
        return true;
    }
}
