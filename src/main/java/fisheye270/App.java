package fisheye270;

import fisheye270.alignment.PairAligner;
import fisheye270.config.AppConfig;
import fisheye270.input.FrameSynchronizer;
import fisheye270.input.VideoReader;
import fisheye270.runtime.PipelineResult;
import fisheye270.runtime.Processor;
import fisheye270.tools.SyntheticWorld;
import fisheye270.visualization.Mosaic;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoWriter;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * CLI for the Java 4-camera fisheye → ~270° cylindrical panorama prototype.
 *
 *   mvn -q exec:java -Dexec.args="synth"
 *   mvn -q exec:java -Dexec.args="phase1"
 *   mvn -q exec:java -Dexec.args="phase2"
 *   mvn -q exec:java -Dexec.args="phase3"
 *   mvn -q exec:java -Dexec.args="serve 9090"
 */
public final class App {
    static {
        OpenCvNative.load();
    }

    public static void main(String[] args) throws Exception {
        String cmd = args.length == 0 ? "serve" : args[0];
        Path configDir = Path.of("config");
        AppConfig cfg = Files.exists(configDir.resolve("cameras.yaml"))
                ? AppConfig.load(configDir) : defaultCfg();
        switch (cmd) {
            case "synth" -> synth(cfg);
            case "phase1" -> phase(cfg, "raw", "output/phase1_raw.jpg", "Phase 1: four synchronized raw views.");
            case "phase2" -> phase2(cfg);
            case "phase3" -> phase(cfg, "undistorted", "output/phase3_undistort_all.jpg",
                    "Phase 3: approximate undistort on all four cameras.");
            case "compare-projections", "compare" -> phase(cfg, "compare_proj", "output/phase4_projections.jpg",
                    "Phase 4: perspective vs cylindrical vs spherical. Recommended: cylindrical.");
            case "align" -> align(cfg);
            case "run" -> run(cfg, args);
            case "serve" -> serve(cfg, args);
            default -> {
                System.err.println("Unknown command: " + cmd);
                System.exit(1);
            }
        }
    }

    static AppConfig defaultCfg() {
        AppConfig cfg = new AppConfig();
        for (String id : AppConfig.CAMERA_IDS) {
            AppConfig.Cam c = new AppConfig.Cam();
            c.id = id;
            c.path = "data/synthetic/" + id.toLowerCase() + ".avi";
            c.pose.yawDeg = switch (id) {
                case "LEFT" -> -90;
                case "RIGHT" -> 90;
                case "REAR" -> 180;
                default -> 0;
            };
            cfg.cameras.put(id, c);
        }
        return cfg;
    }

    static void synth(AppConfig cfg) throws Exception {
        var frames = SyntheticWorld.frames(48, 640, 480);
        var paths = SyntheticWorld.writeVideos(Path.of("data/synthetic"), frames, 15);
        for (var e : paths.entrySet()) {
            cfg.cameras.get(e.getKey()).path = e.getValue().toString();
            System.out.println(e.getKey() + ": " + e.getValue());
        }
    }

    static void ensureSynth(AppConfig cfg) throws Exception {
        boolean missing = false;
        for (AppConfig.Cam c : cfg.cameras.values()) {
            if (c.path == null || !Files.exists(Path.of(c.path))) missing = true;
        }
        if (missing) synth(cfg);
    }

    static FrameSynchronizer openSync(AppConfig cfg, boolean loop) {
        FrameSynchronizer sync = new FrameSynchronizer();
        int opened = 0;
        for (AppConfig.Cam c : cfg.cameras.values()) {
            if (!c.enabled || c.path == null) continue;
            VideoReader r = new VideoReader(c.id, Path.of(c.path), loop);
            if (r.isOpen()) {
                sync.add(r);
                opened++;
            } else {
                System.err.println("Cannot open " + c.path);
                r.close();
            }
        }
        if (opened == 0) {
            var frames = SyntheticWorld.frames(8, 320, 240);
            for (var e : frames.entrySet()) {
                sync.addMemory(new VideoReader.Memory(e.getKey(), e.getValue()));
            }
            System.out.println("Using in-memory synthetic frames.");
        }
        return sync;
    }

    static void phase(AppConfig cfg, String view, String save, String msg) throws Exception {
        ensureSynth(cfg);
        try (FrameSynchronizer sync = openSync(cfg, true)) {
            Processor proc = new Processor(cfg);
            PipelineResult r = proc.process(sync.next());
            Files.createDirectories(Path.of("output"));
            Imgcodecs.imwrite(save, Mosaic.view(proc, r, view));
            System.out.println(msg);
            System.out.println("  present=" + r.raw.keySet() + " missing=" + r.missing);
            System.out.println("  wrote " + save);
        }
    }

    static void phase2(AppConfig cfg) throws Exception {
        ensureSynth(cfg);
        try (FrameSynchronizer sync = openSync(cfg, true)) {
            Processor proc = new Processor(cfg);
            PipelineResult r = proc.process(sync.next());
            Files.createDirectories(Path.of("output"));
            var vis = Processor.sideBySide(r.raw.get("FRONT"), r.undistorted.get("FRONT"),
                    "FRONT RAW", "FRONT PERSPECTIVE UNDISTORT");
            Imgcodecs.imwrite("output/phase2_undistort.jpg", vis);
            var m = proc.model("FRONT", r.raw.get("FRONT"));
            System.out.println("Phase 2: FRONT approximate undistort (rectilinear debug).");
            System.out.printf("  K fx=%.1f fy=%.1f cx=%.1f cy=%.1f [%s]%n",
                    m.intrinsics.fx, m.intrinsics.fy, m.intrinsics.cx, m.intrinsics.cy, m.intrinsics.kSource);
            System.out.println("  D k1..k4=[" + m.intrinsics.k1 + ",0,0,0] [" + m.intrinsics.dSource + "]");
            System.out.println("  FOV=ASSUMED. Stitching uses cylindrical maps, not this view.");
            System.out.println("  wrote output/phase2_undistort.jpg");
        }
    }

    static void align(AppConfig cfg) throws Exception {
        ensureSynth(cfg);
        try (FrameSynchronizer sync = openSync(cfg, false)) {
            Processor proc = new Processor(cfg);
            PipelineResult r = proc.process(sync.next());
            String[][] pairs = {{"LEFT", "FRONT"}, {"FRONT", "RIGHT"}, {"RIGHT", "REAR"}, {"REAR", "LEFT"}};
            Files.createDirectories(Path.of("output/alignment"));
            for (String[] p : pairs) {
                var ia = r.projected.get(p[0]);
                var ib = r.projected.get(p[1]);
                if (ia == null || ib == null) {
                    System.out.println(p[0] + "-" + p[1] + ": skipped");
                    continue;
                }
                Mat overlap = new Mat();
                org.opencv.core.Core.bitwise_and(r.valid.get(p[0]), r.valid.get(p[1]), overlap);
                PairAligner al = new PairAligner();
                var t = al.align(ia, ib, overlap, cfg.orbFeatures, cfg.minInliers);
                String target = "FRONT".equals(p[1]) ? p[0] : ("FRONT".equals(p[0]) ? p[1] : p[0]);
                if (t.confidence > 0) {
                    AppConfig.Cam cam = cfg.cameras.get(target);
                    cam.alignment.dx = t.dx;
                    cam.alignment.dy = t.dy;
                    cam.alignment.rotationDeg = t.rotationDeg;
                    cam.alignment.scale = t.scale;
                }
                System.out.printf("%s-%s method=%s inliers=%d dx=%.2f dy=%.2f%n",
                        p[0], p[1], al.method, t.inliers, t.dx, t.dy);
            }
            System.out.println("Alignment estimated (offline). Runtime uses frozen maps.");
        }
    }

    static void run(AppConfig cfg, String[] args) throws Exception {
        ensureSynth(cfg);
        int frames = 24;
        String video = "output/pano.avi";
        for (int i = 1; i < args.length; i++) {
            if ("--frames".equals(args[i]) && i + 1 < args.length) frames = Integer.parseInt(args[++i]);
            if ("--save-video".equals(args[i]) && i + 1 < args.length) video = args[++i];
            if ("--fov".equals(args[i]) && i + 1 < args.length) cfg.panorama.fovHDeg = Double.parseDouble(args[++i]);
        }
        try (FrameSynchronizer sync = openSync(cfg, true)) {
            Processor proc = new Processor(cfg);
            VideoWriter wr = null;
            Files.createDirectories(Path.of("output"));
            for (int i = 0; i < frames; i++) {
                var bundle = sync.next();
                if (bundle == null) break;
                PipelineResult r = proc.process(bundle);
                if (i == 0 && r.panorama != null) {
                    Imgcodecs.imwrite("output/phase12_pano.jpg", Mosaic.view(proc, r, "stitch"));
                    Imgcodecs.imwrite("output/phase8_seams.jpg", Mosaic.view(proc, r, "seams"));
                    Imgcodecs.imwrite("output/phase5_overlap.jpg", Mosaic.view(proc, r, "overlap"));
                    wr = new VideoWriter(video, VideoWriter.fourcc('M', 'J', 'P', 'G'), 15,
                            new org.opencv.core.Size(r.panorama.cols(), r.panorama.rows()), true);
                }
                if (wr != null && r.panorama != null) wr.write(r.panorama);
                if (i == frames - 1) {
                    System.out.printf("Phase 9–13: stitch fps=%.1f latency=%.1fms missing=%s%n",
                            1000.0 / Math.max(r.totalMs, 1), r.totalMs, r.missing);
                }
            }
            if (wr != null) wr.release();
            System.out.println("wrote " + video);
        }
    }

    static void serve(AppConfig cfg, String[] args) throws Exception {
        ensureSynth(cfg);
        int port = 9090;
        if (args.length >= 2) {
            try { port = Integer.parseInt(args[1]); } catch (NumberFormatException ignored) { }
        }
        // Original CLI: serve [port] [front] [left] [right] [rear]
        if (args.length >= 6) {
            cfg.cameras.get("FRONT").path = args[2];
            cfg.cameras.get("LEFT").path = args[3];
            cfg.cameras.get("RIGHT").path = args[4];
            cfg.cameras.get("REAR").path = args[5];
        }
        FrameSynchronizer sync = openSync(cfg, true);
        Processor proc = new Processor(cfg);
        DebugServer.start(cfg, proc, sync, port);
    }
}
