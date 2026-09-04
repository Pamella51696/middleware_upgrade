package fisheye270.stitching;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Blender {
    public static final Scalar FRONT = new Scalar(255, 180, 80);
    public static final Scalar LEFT = new Scalar(120, 220, 80);
    public static final Scalar RIGHT = new Scalar(80, 180, 255);
    public static final Scalar REAR = new Scalar(255, 120, 200);

    private Blender() {}

    public static Mat featherMask(Mat valid, int blendWidth) {
        Mat bin = new Mat();
        Imgproc.threshold(valid, bin, 1, 255, Imgproc.THRESH_BINARY);
        Mat dist = new Mat();
        Imgproc.distanceTransform(bin, dist, Imgproc.DIST_L2, 3);
        Mat w = new Mat();
        dist.convertTo(w, CvType.CV_32FC1, 1.0 / Math.max(blendWidth, 1.0));
        Core.min(w, new Scalar(1.0), w);
        Mat inv = new Mat();
        Imgproc.threshold(valid, inv, 1, 255, Imgproc.THRESH_BINARY_INV);
        w.setTo(new Scalar(0), inv);
        bin.release();
        dist.release();
        inv.release();
        return w;
    }

    public static Mat hardMask(Mat valid) {
        Mat w = new Mat();
        valid.convertTo(w, CvType.CV_32FC1, 1.0 / 255.0);
        return w;
    }

    public static double[] gainOffset(Mat src, Mat ref, Mat overlapMask) {
        Mat m = new Mat();
        Imgproc.threshold(overlapMask, m, 1, 255, Imgproc.THRESH_BINARY);
        if (Core.countNonZero(m) < 80) {
            m.release();
            return new double[]{1, 1, 1, 0, 0, 0};
        }
        List<Mat> sc = new ArrayList<>();
        List<Mat> rc = new ArrayList<>();
        Core.split(src, sc);
        Core.split(ref, rc);
        double[] g = new double[3];
        double[] o = new double[3];
        for (int c = 0; c < 3; c++) {
            Scalar sm = Core.mean(sc.get(c), m);
            Scalar rm = Core.mean(rc.get(c), m);
            Mat sf = new Mat(), rf = new Mat();
            sc.get(c).convertTo(sf, CvType.CV_32F);
            rc.get(c).convertTo(rf, CvType.CV_32F);
            Mat sm2 = new Mat();
            Core.multiply(sf, sf, sm2);
            double sstd = Math.sqrt(Math.max(Core.mean(sm2, m).val[0] - sm.val[0] * sm.val[0], 1e-3));
            Mat rm2 = new Mat();
            Core.multiply(rf, rf, rm2);
            double rstd = Math.sqrt(Math.max(Core.mean(rm2, m).val[0] - rm.val[0] * rm.val[0], 1e-3));
            g[c] = clamp(rstd / sstd, 1.0 / 2.5, 2.5);
            o[c] = clamp(rm.val[0] - g[c] * sm.val[0], -40, 40);
            sf.release();
            rf.release();
            sm2.release();
            rm2.release();
        }
        m.release();
        return new double[]{g[0], g[1], g[2], o[0], o[1], o[2]};
    }

    public static Mat applyGainOffset(Mat img, double[] go) {
        Mat f = new Mat();
        img.convertTo(f, CvType.CV_32FC3);
        List<Mat> ch = new ArrayList<>();
        Core.split(f, ch);
        for (int c = 0; c < 3; c++) {
            Core.multiply(ch.get(c), new Scalar(go[c]), ch.get(c));
            Core.add(ch.get(c), new Scalar(go[c + 3]), ch.get(c));
        }
        Mat merged = new Mat();
        Core.merge(ch, merged);
        Mat out = new Mat();
        merged.convertTo(out, CvType.CV_8UC3);
        f.release();
        merged.release();
        return out;
    }

    public static Mat composite(List<Mat> layers, List<Mat> weights) {
        Mat acc = Mat.zeros(layers.get(0).size(), CvType.CV_32FC3);
        Mat wsum = Mat.zeros(layers.get(0).size(), CvType.CV_32FC1);
        for (int i = 0; i < layers.size(); i++) {
            Mat lf = new Mat();
            layers.get(i).convertTo(lf, CvType.CV_32FC3);
            Mat w3 = new Mat();
            List<Mat> wc = new ArrayList<>();
            wc.add(weights.get(i));
            wc.add(weights.get(i));
            wc.add(weights.get(i));
            Core.merge(wc, w3);
            Mat mul = new Mat();
            Core.multiply(lf, w3, mul);
            Core.add(acc, mul, acc);
            Core.add(wsum, weights.get(i), wsum);
            lf.release();
            w3.release();
            mul.release();
        }
        Mat safe = new Mat();
        Core.max(wsum, new Scalar(1e-6), safe);
        List<Mat> sc = new ArrayList<>();
        sc.add(safe);
        sc.add(safe);
        sc.add(safe);
        Mat s3 = new Mat();
        Core.merge(sc, s3);
        Mat div = new Mat();
        Core.divide(acc, s3, div);
        Mat out = new Mat();
        div.convertTo(out, CvType.CV_8UC3);
        acc.release();
        wsum.release();
        safe.release();
        s3.release();
        div.release();
        return out;
    }

    public static Mat seamWinner(Map<String, Mat> weights) {
        List<String> ids = new ArrayList<>(weights.keySet());
        Mat vis = Mat.zeros(weights.get(ids.get(0)).size(), CvType.CV_8UC3);
        Mat best = Mat.zeros(vis.size(), CvType.CV_32FC1);
        Mat bestIdx = Mat.zeros(vis.size(), CvType.CV_8UC1);
        for (int i = 0; i < ids.size(); i++) {
            Mat w = weights.get(ids.get(i));
            Mat greater = new Mat();
            Core.compare(w, best, greater, Core.CMP_GT);
            w.copyTo(best, greater);
            bestIdx.setTo(new Scalar(i), greater);
            greater.release();
        }
        Map<String, Scalar> colors = new LinkedHashMap<>();
        colors.put("FRONT", FRONT);
        colors.put("LEFT", LEFT);
        colors.put("RIGHT", RIGHT);
        colors.put("REAR", REAR);
        for (int i = 0; i < ids.size(); i++) {
            Mat eq = new Mat();
            Core.compare(bestIdx, new Scalar(i), eq, Core.CMP_EQ);
            vis.setTo(colors.getOrDefault(ids.get(i), new Scalar(200, 200, 200)), eq);
            eq.release();
        }
        best.release();
        bestIdx.release();
        return vis;
    }

    static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
