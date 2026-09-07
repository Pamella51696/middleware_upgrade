package fisheye270.config;

import fisheye270.calibration.Alignment2D;
import fisheye270.calibration.ApproximateIntrinsics;
import fisheye270.calibration.CameraModel;
import fisheye270.calibration.Pose;
import fisheye270.calibration.Provenance;
import fisheye270.projection.PanoramaSpec;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@SuppressWarnings("unchecked")
public final class AppConfig {
    public static final String[] CAMERA_IDS = {"FRONT", "LEFT", "RIGHT", "REAR"};

    public Path configDir;
    public PanoramaSpec panorama = new PanoramaSpec();
    public Map<String, Cam> cameras = new LinkedHashMap<>();
    public String projectionModel = "cylindrical";
    public double perspectiveBalance = 0.5;
    public int blendWidthPx = 80;
    public String seamMode = "feather";
    public boolean photometric = true;
    public String photoReference = "FRONT";
    public int orbFeatures = 2000;
    public int minInliers = 25;

    public static final class Cam {
        public String id;
        public String path;
        public boolean enabled = true;
        public int width = 640, height = 480;
        public double fovH = 190;
        public Double fx, fy, cx, cy;
        public double k1, k2, k3, k4;
        public Provenance kSource = Provenance.ESTIMATED;
        public Provenance dSource = Provenance.ESTIMATED;
        public Pose pose = new Pose();
        public Alignment2D alignment = new Alignment2D();

        public CameraModel toModel(int frameW, int frameH) {
            CameraModel m = new CameraModel();
            m.id = id;
            m.intrinsics = ApproximateIntrinsics.fromSizeAndFov(
                    frameW, frameH, fovH, fx, fy, cx, cy, k1, k2, k3, k4, kSource, dSource);
            m.pose = pose;
            m.alignment = alignment;
            m.enabled = enabled;
            return m;
        }
    }

    public static AppConfig load(Path dir) throws IOException {
        AppConfig cfg = new AppConfig();
        cfg.configDir = dir;
        Map<String, Object> root = SimpleYaml.load(dir.resolve("cameras.yaml"));
        Map<String, Object> out = (Map<String, Object>) root.get("output");
        if (out != null) {
            cfg.panorama.fovHDeg = d(out.get("fov_horizontal_deg"), 270);
            cfg.panorama.fovVDeg = d(out.get("fov_vertical_deg"), 70);
            cfg.panorama.width = (int) d(out.get("width"), 1920);
            cfg.panorama.height = (int) d(out.get("height"), 540);
            cfg.panorama.yawCenterDeg = d(out.get("yaw_center_deg"), 0);
        }
        Map<String, Object> cams = (Map<String, Object>) root.get("cameras");
        for (String id : CAMERA_IDS) {
            if (cams == null || !cams.containsKey(id)) continue;
            cfg.cameras.put(id, parseCam(id, (Map<String, Object>) cams.get(id)));
        }
        Path proj = dir.resolve("projection.yaml");
        if (Files.exists(proj)) {
            Map<String, Object> p = SimpleYaml.load(proj);
            cfg.projectionModel = str(p.get("model"), "cylindrical");
            cfg.panorama.model = cfg.projectionModel;
            Map<String, Object> persp = (Map<String, Object>) p.get("perspective");
            if (persp != null) cfg.perspectiveBalance = d(persp.get("balance"), 0.5);
        }
        Path st = dir.resolve("stitching.yaml");
        if (Files.exists(st)) {
            Map<String, Object> s = SimpleYaml.load(st);
            Map<String, Object> seam = (Map<String, Object>) s.get("seam");
            if (seam != null) {
                cfg.blendWidthPx = (int) d(seam.get("blend_width_px"), 80);
                cfg.seamMode = str(seam.get("mode"), "feather");
            }
            Map<String, Object> ph = (Map<String, Object>) s.get("photometric");
            if (ph != null) {
                cfg.photometric = !Boolean.FALSE.equals(ph.get("enabled"));
                cfg.photoReference = str(ph.get("reference"), "FRONT");
            }
            Map<String, Object> al = (Map<String, Object>) s.get("alignment");
            if (al != null) {
                cfg.orbFeatures = (int) d(al.get("nfeatures"), 2000);
                cfg.minInliers = (int) d(al.get("min_inliers"), 25);
            }
        }
        return cfg;
    }

    static Cam parseCam(String id, Map<String, Object> n) {
        Cam c = new Cam();
        c.id = id;
        Map<String, Object> src = (Map<String, Object>) n.get("source");
        if (src != null) c.path = str(src.get("path"), null);
        Map<String, Object> img = (Map<String, Object>) n.get("image");
        if (img != null) {
            c.width = (int) d(img.get("width"), 640);
            c.height = (int) d(img.get("height"), 480);
        }
        c.fovH = d(n.get("fov_horizontal_deg"), 190);
        Map<String, Object> intra = (Map<String, Object>) n.get("intrinsics");
        if (intra != null) {
            c.fx = boxed(intra.get("fx"));
            c.fy = boxed(intra.get("fy"));
            c.cx = boxed(intra.get("cx"));
            c.cy = boxed(intra.get("cy"));
            c.kSource = prov(intra.get("source"), Provenance.ESTIMATED);
        }
        Map<String, Object> dist = (Map<String, Object>) n.get("distortion");
        if (dist != null) {
            c.k1 = d(dist.get("k1"), 0);
            c.k2 = d(dist.get("k2"), 0);
            c.k3 = d(dist.get("k3"), 0);
            c.k4 = d(dist.get("k4"), 0);
            c.dSource = prov(dist.get("source"), Provenance.ESTIMATED);
        }
        Map<String, Object> pose = (Map<String, Object>) n.get("pose");
        if (pose != null) {
            c.pose.yawDeg = d(pose.get("yaw_deg"), 0);
            c.pose.pitchDeg = d(pose.get("pitch_deg"), 0);
            c.pose.rollDeg = d(pose.get("roll_deg"), 0);
            c.pose.source = prov(pose.get("source"), Provenance.ASSUMED);
        }
        Map<String, Object> al = (Map<String, Object>) n.get("alignment");
        if (al != null) {
            c.alignment.dx = d(al.get("dx"), 0);
            c.alignment.dy = d(al.get("dy"), 0);
            c.alignment.rotationDeg = d(al.get("rotation_deg"), 0);
            c.alignment.scale = d(al.get("scale"), 1);
        }
        Object en = n.get("enabled");
        c.enabled = en == null || Boolean.TRUE.equals(en);
        return c;
    }

    static double d(Object o, double def) {
        if (o == null) return def;
        if (o instanceof Number n) return n.doubleValue();
        return Double.parseDouble(o.toString());
    }

    static Double boxed(Object o) {
        if (o == null) return null;
        if ("null".equalsIgnoreCase(String.valueOf(o))) return null;
        return d(o, 0);
    }

    static String str(Object o, String def) {
        return o == null ? def : o.toString();
    }

    static Provenance prov(Object o, Provenance def) {
        if (o == null) return def;
        try {
            return Provenance.valueOf(o.toString().trim().toUpperCase());
        } catch (Exception e) {
            return def;
        }
    }

    public List<String> enabledIds() {
        return cameras.values().stream().filter(c -> c.enabled).map(c -> c.id).toList();
    }
}
