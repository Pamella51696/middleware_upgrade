package fisheye270.projection;

import fisheye270.calibration.CameraModel;
import fisheye270.calibration.FisheyeModel;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Precomputed dest-pano → source-fisheye LUTs.
 *
 * pixel (x,y) → theta, phi → vehicle ray → camera ray → Kannala–Brandt (u,v)
 */
public final class ProjectionMaps {
    public final Mat mapX;
    public final Mat mapY;
    public final Mat valid;

    public ProjectionMaps(Mat mapX, Mat mapY, Mat valid) {
        this.mapX = mapX;
        this.mapY = mapY;
        this.valid = valid;
    }

    public void release() {
        mapX.release();
        mapY.release();
        valid.release();
    }

    public static ProjectionMaps build(CameraModel cam, PanoramaSpec spec) {
        int w = spec.width;
        int h = spec.height;
        Mat mapX = new Mat(h, w, CvType.CV_32FC1);
        Mat mapY = new Mat(h, w, CvType.CV_32FC1);
        Mat valid = new Mat(h, w, CvType.CV_8UC1);
        float[] rowX = new float[w];
        float[] rowY = new float[w];
        byte[] rowV = new byte[w];
        float[] uv = new float[2];
        double[] rayC = new double[3];
        double theta0 = spec.theta0();
        double thetaSpan = spec.theta1() - spec.theta0();
        double fovV = Math.toRadians(spec.fovVDeg);
        boolean spherical = "spherical".equals(spec.model);

        for (int y = 0; y < h; y++) {
            double phi = ((double) y / Math.max(h - 1, 1) - 0.5) * fovV;
            for (int x = 0; x < w; x++) {
                double theta = theta0 + ((double) x / Math.max(w - 1, 1)) * thetaSpan;
                double dx, dy, dz;
                if (spherical) {
                    dx = Math.sin(theta) * Math.cos(phi);
                    dy = Math.sin(phi);
                    dz = Math.cos(theta) * Math.cos(phi);
                } else {
                    dx = Math.sin(theta);
                    dy = Math.tan(phi);
                    dz = Math.cos(theta);
                    double n = Math.sqrt(dx * dx + dy * dy + dz * dz);
                    dx /= n;
                    dy /= n;
                    dz /= n;
                }
                cam.vehicleToCamera(dx, dy, dz, rayC);
                boolean ok = FisheyeModel.project(cam.intrinsics, rayC[0], rayC[1], rayC[2], uv);
                if (ok) {
                    rowX[x] = uv[0];
                    rowY[x] = uv[1];
                    rowV[x] = (byte) 255;
                } else {
                    rowX[x] = -1f;
                    rowY[x] = -1f;
                    rowV[x] = 0;
                }
            }
            mapX.put(y, 0, rowX);
            mapY.put(y, 0, rowY);
            valid.put(y, 0, rowV);
        }
        applyAlignment(mapX, mapY, cam);
        return new ProjectionMaps(mapX, mapY, valid);
    }

    static void applyAlignment(Mat mapX, Mat mapY, CameraModel cam) {
        var a = cam.alignment;
        if (Math.abs(a.dx) < 1e-6 && Math.abs(a.dy) < 1e-6
                && Math.abs(a.rotationDeg) < 1e-6 && Math.abs(a.scale - 1.0) < 1e-6) {
            return;
        }
        int h = mapX.rows();
        int w = mapX.cols();
        double angle = Math.toRadians(a.rotationDeg);
        double s = Math.max(a.scale, 1e-4);
        double ca = Math.cos(angle), sa = Math.sin(angle);
        double cx = (w - 1) / 2.0, cy = (h - 1) / 2.0;
        Mat qx = new Mat(h, w, CvType.CV_32FC1);
        Mat qy = new Mat(h, w, CvType.CV_32FC1);
        float[] rx = new float[w];
        float[] ry = new float[w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double px = x - cx - a.dx;
                double py = y - cy - a.dy;
                rx[x] = (float) ((ca * px + sa * py) / s + cx);
                ry[x] = (float) ((-sa * px + ca * py) / s + cy);
            }
            qx.put(y, 0, rx);
            qy.put(y, 0, ry);
        }
        Mat nx = new Mat();
        Mat ny = new Mat();
        Imgproc.remap(mapX, nx, qx, qy, Imgproc.INTER_LINEAR);
        Imgproc.remap(mapY, ny, qx, qy, Imgproc.INTER_LINEAR);
        nx.copyTo(mapX);
        ny.copyTo(mapY);
        nx.release();
        ny.release();
        qx.release();
        qy.release();
    }
}
