package fisheye270;

import fisheye270.calibration.ApproximateIntrinsics;
import fisheye270.calibration.Provenance;
import fisheye270.config.AppConfig;
import fisheye270.input.FrameSynchronizer;
import fisheye270.input.VideoReader;
import fisheye270.runtime.PipelineResult;
import fisheye270.runtime.Processor;
import fisheye270.tools.SyntheticWorld;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PipelineTest {
    @BeforeAll
    static void loadCv() {
        OpenCvNative.load();
    }

    @Test
    void intrinsicsFromFov() {
        var i = ApproximateIntrinsics.fromSizeAndFov(
                640, 480, 180, null, null, null, null, 0, 0, 0, 0,
                Provenance.ESTIMATED, Provenance.ESTIMATED);
        assertEquals(320.0, i.cx);
        assertEquals(320.0 / (Math.PI / 2), i.fx, 1e-6);
        assertEquals(Provenance.ESTIMATED, i.kSource);
    }

    @Test
    void phase1SyncFourCameras() {
        Map<String, List<Mat>> frames = SyntheticWorld.frames(4, 320, 240);
        FrameSynchronizer sync = new FrameSynchronizer();
        for (var e : frames.entrySet()) {
            sync.addMemory(new VideoReader.Memory(e.getKey(), e.getValue()));
        }
        var bundle = sync.next();
        assertEquals(4, bundle.frames.size());
        assertTrue(bundle.missing.isEmpty());
    }

    @Test
    void undistortAndCylindricalPano() {
        AppConfig cfg = App.defaultCfg();
        cfg.panorama.width = 960;
        cfg.panorama.height = 240;
        cfg.panorama.fovHDeg = 270;
        Processor proc = new Processor(cfg);
        Map<String, List<Mat>> frames = SyntheticWorld.frames(3, 320, 240);
        FrameSynchronizer sync = new FrameSynchronizer();
        for (var e : frames.entrySet()) {
            sync.addMemory(new VideoReader.Memory(e.getKey(), e.getValue()));
        }
        PipelineResult r = proc.process(sync.next());
        assertEquals(4, r.undistorted.size());
        assertNotNull(r.panorama);
        assertTrue(r.panorama.cols() > r.panorama.rows());
        assertEquals("cylindrical", proc.spec.model);
        org.opencv.core.MatOfDouble mean = new org.opencv.core.MatOfDouble();
        org.opencv.core.MatOfDouble std = new org.opencv.core.MatOfDouble();
        org.opencv.core.Core.meanStdDev(r.panorama, mean, std);
        assertTrue(mean.get(0, 0)[0] > 5, "panorama should not be empty");
    }

    @Test
    void yamlConfigLoadsWithoutSnakeYaml() throws Exception {
        var cfg = AppConfig.load(java.nio.file.Path.of("config"));
        assertEquals(4, cfg.cameras.size());
        assertEquals(270.0, cfg.panorama.fovHDeg);
        assertEquals("cylindrical", cfg.projectionModel);
        assertNotNull(cfg.cameras.get("FRONT").path);
        assertEquals(-90.0, cfg.cameras.get("LEFT").pose.yawDeg);
    }

    @Test
    void missingCameraContinues() {
        AppConfig cfg = App.defaultCfg();
        Processor proc = new Processor(cfg);
        Map<String, List<Mat>> frames = SyntheticWorld.frames(2, 320, 240);
        FrameSynchronizer sync = new FrameSynchronizer();
        for (var e : frames.entrySet()) {
            if (!"REAR".equals(e.getKey())) {
                sync.addMemory(new VideoReader.Memory(e.getKey(), e.getValue()));
            }
        }
        PipelineResult r = proc.process(sync.next());
        assertTrue(r.missing.contains("REAR"));
        assertNotNull(r.panorama);
        assertFalse(r.panorama.empty());
    }
}
