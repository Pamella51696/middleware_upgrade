package fisheye270.input;

import org.opencv.core.Mat;

public final class Frame {
    public final String cameraId;
    public final Mat image;
    public final boolean ok;
    public final int index;

    public Frame(String cameraId, Mat image, boolean ok, int index) {
        this.cameraId = cameraId;
        this.image = image;
        this.ok = ok;
        this.index = index;
    }
}
