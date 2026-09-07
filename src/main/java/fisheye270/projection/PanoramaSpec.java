package fisheye270.projection;

/** Common panoramic canvas. Not a bird's-eye / ground-plane coordinate system. */
public final class PanoramaSpec {
    public int width = 1920;
    public int height = 540;
    public double fovHDeg = 270;
    public double fovVDeg = 70;
    public double yawCenterDeg = 0;
    /** cylindrical | spherical | perspective */
    public String model = "cylindrical";

    public double theta0() {
        return Math.toRadians(yawCenterDeg - fovHDeg / 2.0);
    }

    public double theta1() {
        return Math.toRadians(yawCenterDeg + fovHDeg / 2.0);
    }
}
