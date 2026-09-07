package fisheye270.alignment;

import fisheye270.calibration.Alignment2D;
import fisheye270.calibration.Provenance;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Offline ORB + RANSAC similarity. Homography is not used: cameras have
 * parallax and the scene is not a plane.
 */
public final class PairAligner {
    public final Alignment2D transform = new Alignment2D();
    public String method = "orb+ransac";

    public Alignment2D align(Mat imgA, Mat imgB, Mat overlap, int nfeatures, int minInliers) {
        Mat grayA = new Mat(), grayB = new Mat();
        Imgproc.cvtColor(imgA, grayA, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(imgB, grayB, Imgproc.COLOR_BGR2GRAY);
        ORB orb = ORB.create(nfeatures);
        MatOfKeyPoint kpA = new MatOfKeyPoint();
        MatOfKeyPoint kpB = new MatOfKeyPoint();
        Mat dA = new Mat(), dB = new Mat();
        orb.detectAndCompute(grayA, overlap, kpA, dA);
        orb.detectAndCompute(grayB, overlap, kpB, dB);
        transform.source = Provenance.ASSUMED;
        if (dA.empty() || dB.empty() || dA.rows() < 8 || dB.rows() < 8) {
            method = "hold-identity";
            return transform;
        }
        BFMatcher bf = BFMatcher.create(org.opencv.core.Core.NORM_HAMMING, false);
        List<MatOfDMatch> knn = new ArrayList<>();
        bf.knnMatch(dA, dB, knn, 2);
        List<DMatch> good = new ArrayList<>();
        for (MatOfDMatch mm : knn) {
            DMatch[] arr = mm.toArray();
            if (arr.length < 2) continue;
            if (arr[0].distance < 0.75 * arr[1].distance) good.add(arr[0]);
        }
        if (good.size() < 8) {
            method = "hold-identity";
            return transform;
        }
        Point[] pa = new Point[good.size()];
        Point[] pb = new Point[good.size()];
        var kpa = kpA.toArray();
        var kpb = kpB.toArray();
        for (int i = 0; i < good.size(); i++) {
            pa[i] = kpa[good.get(i).queryIdx].pt;
            pb[i] = kpb[good.get(i).trainIdx].pt;
        }
        MatOfPoint2f paM = new MatOfPoint2f(pa);
        MatOfPoint2f pbM = new MatOfPoint2f(pb);
        Mat inliers = new Mat();
        Mat M = Calib3d.estimateAffinePartial2D(paM, pbM, inliers, Calib3d.RANSAC, 4.0, 2000, 0.99, 10);
        if (M.empty()) {
            method = "hold-identity";
            return transform;
        }
        double a = M.get(0, 0)[0];
        double b = M.get(1, 0)[0];
        transform.scale = Math.hypot(a, b);
        transform.rotationDeg = Math.toDegrees(Math.atan2(b, a));
        transform.dx = M.get(0, 2)[0];
        transform.dy = M.get(1, 2)[0];
        transform.inliers = inliers.empty() ? 0 : (int) org.opencv.core.Core.countNonZero(inliers);
        transform.confidence = transform.inliers / (double) Math.max(good.size(), 1);
        transform.source = Provenance.ESTIMATED;
        boolean unreasonable = Math.abs(transform.dx) > 120 || Math.abs(transform.dy) > 80
                || Math.abs(transform.rotationDeg) > 8 || Math.abs(transform.scale - 1) > 0.15;
        if (transform.inliers < minInliers || transform.confidence < 0.25 || unreasonable) {
            method = "hold-identity";
            transform.dx = 0;
            transform.dy = 0;
            transform.rotationDeg = 0;
            transform.scale = 1;
            transform.confidence = 0;
            transform.source = Provenance.ASSUMED;
        } else {
            method = "orb+ransac";
        }
        return transform;
    }
}
