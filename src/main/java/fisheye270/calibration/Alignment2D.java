package fisheye270.calibration;

/** Similarity in panorama pixels. Frozen at runtime. */
public final class Alignment2D {
    public double dx;
    public double dy;
    public double rotationDeg;
    public double scale = 1.0;
    public Provenance source = Provenance.ESTIMATED;
    public double confidence;
    public int inliers;
}
