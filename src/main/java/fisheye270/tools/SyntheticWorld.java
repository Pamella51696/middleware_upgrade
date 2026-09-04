package fisheye270.tools;

import fisheye270.calibration.ApproximateIntrinsics;
import fisheye270.calibration.CameraModel;
import fisheye270.calibration.Provenance;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoWriter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Stand-in overlapping fisheye cameras until real FRONT/LEFT/RIGHT/REAR files exist. */
public final class SyntheticWorld {
    public static Mat makeWorld(int width, int height) {
        Mat img = new Mat(height, width, CvType.CV_8UC3);
        img.rowRange(0, height / 2).setTo(new Scalar(90, 170, 210));
        img.rowRange(height / 2, height).setTo(new Scalar(50, 50, 50));
        int horizon = height / 2;
        Imgproc.line(img, new Point(0, horizon), new Point(width, horizon), new Scalar(240, 240, 240), 3);
        for (int deg = -180; deg < 180; deg += 10) {
            int x = (int) ((deg + 180) / 360.0 * width);
            Scalar color = new Scalar(40 + Math.abs(deg) % 180, 80, 200 - Math.abs(deg) % 120);
            if (deg % 30 == 0) {
                Imgproc.rectangle(img, new Point(x - 18, horizon - 180), new Point(x + 18, horizon), color, -1);
                Imgproc.putText(img, Integer.toString(deg), new Point(x - 24, horizon - 190),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);
            }
            Imgproc.line(img, new Point(x, 0), new Point(x, height), new Scalar(30, 30, 30), 1);
        }
        for (int x = 0; x < width; x += 40) {
            Imgproc.rectangle(img, new Point(x, horizon + 80), new Point(x + 20, horizon + 92),
                    new Scalar(200, 200, 200), -1);
        }
        Imgproc.putText(img, "FRONT 0 deg", new Point(width / 2.0 - 80, 80),
                Imgproc.FONT_HERSHEY_SIMPLEX, 1.2, new Scalar(0, 0, 0), 3);
        return img;
    }

    public static CameraModel camera(String id, int w, int h, double yaw) {
        CameraModel m = new CameraModel();
        m.id = id;
        m.intrinsics = ApproximateIntrinsics.fromSizeAndFov(
                w, h, 190, null, null, null, null, 0, 0, 0, 0,
                Provenance.ESTIMATED, Provenance.ESTIMATED);
        m.pose.yawDeg = yaw;
        return m;
    }

    public static Mat renderFisheye(Mat world, CameraModel cam) {
        int h = cam.intrinsics.height;
        int w = cam.intrinsics.width;
        Mat mapX = new Mat(h, w, CvType.CV_32FC1);
        Mat mapY = new Mat(h, w, CvType.CV_32FC1);
        float[] rx = new float[w];
        float[] ry = new float[w];
        double[][] R = cam.rotationCameraToVehicle();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double a = (x - cam.intrinsics.cx) / cam.intrinsics.fx;
                double b = (y - cam.intrinsics.cy) / cam.intrinsics.fy;
                double r = Math.hypot(a, b);
                double theta = r;
                for (int i = 0; i < 4; i++) {
                    double t2 = theta * theta;
                    double k = 1 + cam.intrinsics.k1 * t2;
                    theta = k > 1e-8 ? r / k : theta;
                }
                double zc = Math.cos(theta);
                double s = Math.sin(theta);
                double xc = r > 1e-8 ? a * s / r : 0;
                double yc = r > 1e-8 ? b * s / r : 0;
                // camera → vehicle: ray_v = R * ray_c
                double vx = R[0][0] * xc + R[0][1] * yc + R[0][2] * zc;
                double vy = R[1][0] * xc + R[1][1] * yc + R[1][2] * zc;
                double vz = R[2][0] * xc + R[2][1] * yc + R[2][2] * zc;
                double th = Math.atan2(vx, vz);
                double hyp = Math.hypot(vx, vz);
                double phi = Math.atan2(vy, Math.max(hyp, 1e-8));
                rx[x] = (float) ((th + Math.PI) / (2 * Math.PI) * (world.cols() - 1));
                ry[x] = (float) Math.max(0, Math.min(world.rows() - 1,
                        (phi + Math.toRadians(50)) / Math.toRadians(100) * (world.rows() - 1)));
            }
            mapX.put(y, 0, rx);
            mapY.put(y, 0, ry);
        }
        Mat out = new Mat();
        Imgproc.remap(world, out, mapX, mapY, Imgproc.INTER_LINEAR, Core.BORDER_WRAP);
        mapX.release();
        mapY.release();
        return out;
    }

    public static Map<String, List<Mat>> frames(int n, int w, int h) {
        Mat world = makeWorld(4096, 1024);
        Map<String, Double> yaws = new LinkedHashMap<>();
        yaws.put("FRONT", 0.0);
        yaws.put("LEFT", -90.0);
        yaws.put("RIGHT", 90.0);
        yaws.put("REAR", 180.0);
        Map<String, CameraModel> cams = new LinkedHashMap<>();
        for (var e : yaws.entrySet()) cams.put(e.getKey(), camera(e.getKey(), w, h, e.getValue()));
        Map<String, List<Mat>> out = new LinkedHashMap<>();
        for (String id : cams.keySet()) out.put(id, new ArrayList<>());
        for (int i = 0; i < n; i++) {
            Mat rolled = rollX(world, i * 6);
            for (var e : cams.entrySet()) {
                Mat img = renderFisheye(rolled, e.getValue());
                if ("RIGHT".equals(e.getKey())) {
                    img.convertTo(img, -1, 1.15, 8);
                }
                out.get(e.getKey()).add(img);
            }
            rolled.release();
        }
        return out;
    }

    static Mat rollX(Mat src, int shift) {
        int w = src.cols();
        shift = ((shift % w) + w) % w;
        if (shift == 0) return src.clone();
        Mat a = src.colRange(shift, w);
        Mat b = src.colRange(0, shift);
        Mat out = new Mat();
        Core.hconcat(List.of(a, b), out);
        return out;
    }

    public static Map<String, Path> writeVideos(Path dir, Map<String, List<Mat>> frames, double fps) throws Exception {
        Files.createDirectories(dir);
        Map<String, Path> paths = new LinkedHashMap<>();
        for (var e : frames.entrySet()) {
            Path p = dir.resolve(e.getKey().toLowerCase() + ".avi");
            Mat first = e.getValue().get(0);
            int fourcc = VideoWriter.fourcc('M', 'J', 'P', 'G');
            VideoWriter wr = new VideoWriter(p.toString(), fourcc, fps,
                    new Size(first.cols(), first.rows()), true);
            if (!wr.isOpened()) {
                throw new IllegalStateException("VideoWriter failed for " + p);
            }
            for (Mat im : e.getValue()) wr.write(im);
            wr.release();
            paths.put(e.getKey(), p);
        }
        return paths;
    }
}
