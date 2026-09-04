package fisheye270.input;

import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import java.nio.file.Path;
import java.util.List;

public final class VideoReader implements AutoCloseable {
    public final String cameraId;
    private VideoCapture cap;
    private final Path path;
    private final boolean loop;
    private int index;

    public VideoReader(String cameraId, Path path, boolean loop) {
        this.cameraId = cameraId;
        this.path = path;
        this.loop = loop;
        this.cap = new VideoCapture(path.toString());
    }

    public boolean isOpen() {
        return cap != null && cap.isOpened();
    }

    public Frame read() {
        if (!isOpen()) {
            return new Frame(cameraId, new Mat(), false, index);
        }
        Mat img = new Mat();
        boolean ok = cap.read(img);
        if (!ok || img.empty()) {
            if (loop) {
                cap.release();
                cap = new VideoCapture(path.toString());
                ok = cap.read(img);
            }
        }
        if (!ok || img.empty()) {
            return new Frame(cameraId, img, false, index);
        }
        return new Frame(cameraId, img, true, index++);
    }

    /** In-memory repeating source used by tests. */
    public static class Memory {
        public final String cameraId;
        private final List<Mat> frames;
        private int i;

        public Memory(String cameraId, List<Mat> frames) {
            this.cameraId = cameraId;
            this.frames = frames;
        }

        public Frame read() {
            Mat src = frames.get(i % frames.size());
            i++;
            return new Frame(cameraId, src, true, i - 1);
        }
    }

    @Override
    public void close() {
        if (cap != null) {
            cap.release();
            cap = null;
        }
    }
}
