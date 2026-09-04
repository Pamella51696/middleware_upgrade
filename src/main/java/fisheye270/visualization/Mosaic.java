package fisheye270.visualization;

import fisheye270.runtime.PipelineResult;
import fisheye270.runtime.Processor;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public final class Mosaic {
    private Mosaic() {}

    public static Mat view(Processor proc, PipelineResult r, String view) {
        return switch (view) {
            case "undistorted" -> proc.labeledQuad(r.undistorted, "View 2 - PERSPECTIVE UNDISTORT (debug)");
            case "projected" -> proc.labeledQuad(r.projected, "View 3 - " + proc.spec.model.toUpperCase());
            case "overlap" -> overlap(r);
            case "stitch" -> caption(orEmpty(r.panorama),
                    "View 7 - STITCH fov=" + (int) proc.spec.fovHDeg + " missing=" +
                            (r.missing.isEmpty() ? "none" : String.join(",", r.missing)));
            case "seams" -> caption(orEmpty(r.seams), "View 8 - SEAM WINNER");
            case "compare_proj" -> compare(proc, r);
            default -> proc.labeledQuad(r.raw, "View 1 - RAW");
        };
    }

    static Mat overlap(PipelineResult r) {
        List<Mat> rows = new ArrayList<>();
        String[][] pairs = {{"LEFT", "FRONT"}, {"FRONT", "RIGHT"}};
        for (String[] p : pairs) {
            Mat a = r.projected.get(p[0]);
            Mat b = r.projected.get(p[1]);
            if (a == null || b == null) continue;
            Mat blend = new Mat();
            Core.addWeighted(a, 0.5, b, 0.5, 0, blend);
            Mat small = new Mat();
            Imgproc.resize(blend, small, new Size(960, 270));
            Imgproc.putText(small, "View 4 - overlap " + p[0] + " + " + p[1],
                    new Point(10, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
            rows.add(small);
        }
        if (rows.isEmpty()) return Mat.zeros(270, 960, CvType.CV_8UC3);
        Mat out = new Mat();
        Core.vconcat(rows, out);
        return out;
    }

    static Mat compare(Processor proc, PipelineResult r) {
        Mat raw = r.raw.get("FRONT");
        if (raw == null) return Mat.zeros(240, 720, CvType.CV_8UC3);
        List<Mat> panels = new ArrayList<>();
        String saved = proc.spec.model;
        for (String name : new String[]{"perspective", "cylindrical", "spherical"}) {
            Mat img;
            if ("perspective".equals(name)) {
                img = r.undistorted.getOrDefault("FRONT", raw);
            } else {
                proc.spec.model = name;
                proc.invalidate("FRONT");
                img = proc.project("FRONT", raw)[0];
            }
            Mat small = new Mat();
            Imgproc.resize(img, small, new Size(640, 180));
            Imgproc.putText(small, name, new Point(10, 28),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
            panels.add(small);
        }
        proc.spec.model = saved;
        proc.invalidate("FRONT");
        Mat out = new Mat();
        Core.vconcat(panels, out);
        return out;
    }

    static Mat orEmpty(Mat m) {
        return m == null || m.empty() ? Mat.zeros(200, 400, CvType.CV_8UC3) : m;
    }

    static Mat caption(Mat img, String text) {
        Mat out = img.clone();
        Imgproc.putText(out, text, new Point(10, 28),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 0, 0), 4);
        Imgproc.putText(out, text, new Point(10, 28),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);
        return out;
    }
}
