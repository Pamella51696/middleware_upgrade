package fisheye270.input;

import fisheye270.config.AppConfig;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Lock-step sync for prerecorded files (no shared clock).
 * A dead camera is omitted; remaining views still composite.
 */
public final class FrameSynchronizer implements AutoCloseable {
    private final Map<String, VideoReader> files = new LinkedHashMap<>();
    private final Map<String, VideoReader.Memory> memory = new LinkedHashMap<>();
    private int tick;

    public void add(VideoReader r) {
        files.put(r.cameraId, r);
    }

    public void addMemory(VideoReader.Memory m) {
        memory.put(m.cameraId, m);
    }

    public Bundle next() {
        Map<String, Frame> got = new LinkedHashMap<>();
        List<String> missing = new ArrayList<>();
        for (String id : AppConfig.CAMERA_IDS) {
            Frame fr = null;
            if (files.containsKey(id)) {
                fr = files.get(id).read();
            } else if (memory.containsKey(id)) {
                fr = memory.get(id).read();
            } else {
                missing.add(id);
                continue;
            }
            if (fr == null || !fr.ok) {
                missing.add(id);
                continue;
            }
            got.put(id, fr);
        }
        if (got.isEmpty()) return null;
        return new Bundle(got, missing, tick++);
    }

    @Override
    public void close() {
        for (VideoReader r : files.values()) r.close();
    }

    public static final class Bundle {
        public final Map<String, Frame> frames;
        public final List<String> missing;
        public final int index;

        public Bundle(Map<String, Frame> frames, List<String> missing, int index) {
            this.frames = frames;
            this.missing = missing;
            this.index = index;
        }

        public Mat image(String id) {
            Frame f = frames.get(id);
            return (f != null && f.ok) ? f.image : null;
        }
    }
}
