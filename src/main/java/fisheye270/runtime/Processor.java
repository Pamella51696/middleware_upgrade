package fisheye270.runtime;

import fisheye270.calibration.CameraModel;
import fisheye270.calibration.FisheyeModel;
import fisheye270.config.AppConfig;
import fisheye270.input.FrameSynchronizer;
import fisheye270.projection.PanoramaSpec;
import fisheye270.projection.ProjectionMaps;
import fisheye270.stitching.Blender;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class Processor {
    public final AppConfig cfg;
    public final PanoramaSpec spec;
    private final Map<String, CameraModel> models = new LinkedHashMap<>();
    private final Map<String, ProjectionMaps> panoMaps = new LinkedHashMap<>();
    private final Map<String, Mat[]> undistMaps = new LinkedHashMap<>();
    private final Map<String, Mat> weightCache = new LinkedHashMap<>();
    private final Map<String, double[]> photo = new LinkedHashMap<>();
    private Mat seams;

    public Processor(AppConfig cfg) {
        this.cfg = cfg;
        this.spec = cfg.panorama;
        this.spec.model = cfg.projectionModel;
    }

    public CameraModel model(String id, Mat frame) {
        CameraModel existing = models.get(id);
        if (existing != null
                && existing.intrinsics.width == frame.cols()
                && existing.intrinsics.height == frame.rows()) {
            return existing;
        }
        CameraModel m = cfg.cameras.get(id).toModel(frame.cols(), frame.rows());
        models.put(id, m);
        panoMaps.remove(id);
        undistMaps.remove(id);
        weightCache.clear();
        seams = null;
        return m;
    }

    public void invalidate(String id) {
        models.remove(id);
        panoMaps.remove(id);
        undistMaps.remove(id);
        weightCache.clear();
        photo.clear();
        seams = null;
    }

    public Mat undistort(String id, Mat frame) {
        CameraModel m = model(id, frame);
        Mat[] maps = undistMaps.get(id);
        if (maps == null) {
            maps = FisheyeModel.undistortMaps(m.intrinsics, cfg.perspectiveBalance, 1.0);
            undistMaps.put(id, maps);
        }
        return FisheyeModel.remap(frame, maps[0], maps[1]);
    }

    public Mat[] project(String id, Mat frame) {
        CameraModel m = model(id, frame);
        ProjectionMaps maps = panoMaps.get(id);
        if (maps == null) {
            maps = ProjectionMaps.build(m, spec);
            panoMaps.put(id, maps);
        }
        return new Mat[]{FisheyeModel.remap(frame, maps.mapX, maps.mapY), maps.valid};
    }

    public PipelineResult process(FrameSynchronizer.Bundle bundle) {
        long t0 = System.nanoTime();
        PipelineResult r = new PipelineResult();
        r.index = bundle.index;
        r.missing.addAll(bundle.missing);
        for (String id : AppConfig.CAMERA_IDS) {
            Mat img = bundle.image(id);
            if (img == null) continue;
            r.raw.put(id, img);
            r.undistorted.put(id, undistort(id, img));
            Mat[] pv = project(id, img);
            r.projected.put(id, pv[0]);
            r.valid.put(id, pv[1]);
        }
        stitch(r);
        r.totalMs = (System.nanoTime() - t0) / 1e6;
        return r;
    }

    void stitch(PipelineResult r) {
        if (r.projected.isEmpty()) return;
        List<String> order = new ArrayList<>();
        for (String id : new String[]{"LEFT", "FRONT", "RIGHT"}) {
            if (r.projected.containsKey(id)) order.add(id);
        }
        if (spec.fovHDeg > 270.5 && r.projected.containsKey("REAR")) {
            order.add("REAR");
        }
        List<Mat> layers = new ArrayList<>();
        List<Mat> weights = new ArrayList<>();
        Map<String, Mat> wmap = new LinkedHashMap<>();
        Mat ref = r.projected.get(cfg.photoReference);
        Mat refV = r.valid.get(cfg.photoReference);
        for (String id : order) {
            Mat img = r.projected.get(id);
            Mat val = r.valid.get(id);
            if (cfg.photometric && ref != null && !id.equals(cfg.photoReference)) {
                double[] go = photo.get(id);
                if (go == null) {
                    Mat overlap = new Mat();
                    if (refV != null) {
                        Core.bitwise_and(val, refV, overlap);
                    } else {
                        overlap = val;
                    }
                    go = Blender.gainOffset(img, ref, overlap);
                    photo.put(id, go);
                }
                img = Blender.applyGainOffset(img, go);
            }
            Mat wt = weightCache.get(id);
            if (wt == null) {
                wt = "hard".equals(cfg.seamMode)
                        ? Blender.hardMask(val)
                        : Blender.featherMask(val, cfg.blendWidthPx);
                weightCache.put(id, wt);
            }
            layers.add(img);
            weights.add(wt);
            wmap.put(id, wt);
        }
        r.panorama = Blender.composite(layers, weights);
        if (seams == null) seams = Blender.seamWinner(wmap);
        r.seams = seams;
    }

    public Mat labeledQuad(Map<String, Mat> images, String title) {
        Mat[] tiles = new Mat[4];
        String[] ids = AppConfig.CAMERA_IDS;
        for (int i = 0; i < 4; i++) {
            Mat src = images.get(ids[i]);
            if (src == null || src.empty()) {
                tiles[i] = Mat.zeros(180, 320, org.opencv.core.CvType.CV_8UC3);
            } else {
                Mat small = new Mat();
                double scale = 320.0 / src.cols();
                Imgproc.resize(src, small, new Size(320, Math.max(1, src.rows() * scale)));
                tiles[i] = small;
            }
            Imgproc.putText(tiles[i], ids[i], new Point(8, 24),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 255), 2);
        }
        int th = 0, tw = 0;
        for (Mat t : tiles) {
            th = Math.max(th, t.rows());
            tw = Math.max(tw, t.cols());
        }
        for (int i = 0; i < 4; i++) {
            if (tiles[i].rows() != th || tiles[i].cols() != tw) {
                Mat pad = Mat.zeros(th, tw, org.opencv.core.CvType.CV_8UC3);
                tiles[i].copyTo(pad.submat(0, tiles[i].rows(), 0, tiles[i].cols()));
                tiles[i] = pad;
            }
        }
        Mat top = new Mat(), bot = new Mat(), out = new Mat();
        Core.hconcat(List.of(tiles[0], tiles[1]), top);
        Core.hconcat(List.of(tiles[2], tiles[3]), bot);
        Core.vconcat(List.of(top, bot), out);
        Imgproc.putText(out, title, new Point(10, 28),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);
        return out;
    }

    public static Mat sideBySide(Mat a, Mat b, String la, String lb) {
        Mat fa = fitH(a, 480);
        Mat fb = fitH(b, 480);
        Imgproc.putText(fa, la, new Point(10, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);
        Imgproc.putText(fb, lb, new Point(10, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);
        Mat out = new Mat();
        Core.hconcat(List.of(fa, fb), out);
        return out;
    }

    static Mat fitH(Mat im, int h) {
        Mat o = new Mat();
        double s = h / (double) im.rows();
        Imgproc.resize(im, o, new Size(Math.max(1, im.cols() * s), h));
        return o;
    }
}
