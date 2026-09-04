package fisheye270.calibration;

/** Vehicle-frame pose. tx/ty/tz are UNKNOWN → zero-parallax prototype. */
public final class Pose {
    public double yawDeg;
    public double pitchDeg;
    public double rollDeg;
    public double tx, ty, tz;
    public Provenance source = Provenance.ASSUMED;
}
