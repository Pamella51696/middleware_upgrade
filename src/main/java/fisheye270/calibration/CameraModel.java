package fisheye270.calibration;

/**
 * Camera look (+Z) in vehicle frame (OpenCV: X right, Y down, Z forward).
 * Columns of R are camera X,Y,Z expressed in the vehicle frame.
 */
public final class CameraModel {
    public String id;
    public Intrinsics intrinsics = new Intrinsics();
    public Pose pose = new Pose();
    public Alignment2D alignment = new Alignment2D();
    public boolean enabled = true;

    public double[][] rotationCameraToVehicle() {
        double yaw = Math.toRadians(pose.yawDeg);
        double pitch = Math.toRadians(pose.pitchDeg);
        double roll = Math.toRadians(pose.rollDeg);
        double[] look = {
            Math.sin(yaw) * Math.cos(pitch),
            Math.sin(pitch),
            Math.cos(yaw) * Math.cos(pitch)
        };
        double[] downWorld = {0, 1, 0};
        double[] right = cross(downWorld, look);
        double n = norm(right);
        if (n < 1e-8) {
            right = new double[]{1, 0, 0};
        } else {
            scaleInPlace(right, 1.0 / n);
        }
        double[] down = cross(look, right);
        scaleInPlace(down, 1.0 / (norm(down) + 1e-12));
        double cr = Math.cos(roll), sr = Math.sin(roll);
        double[] rightR = {
            cr * right[0] + sr * down[0],
            cr * right[1] + sr * down[1],
            cr * right[2] + sr * down[2]
        };
        double[] downR = {
            -sr * right[0] + cr * down[0],
            -sr * right[1] + cr * down[1],
            -sr * right[2] + cr * down[2]
        };
        return new double[][]{
            {rightR[0], downR[0], look[0]},
            {rightR[1], downR[1], look[1]},
            {rightR[2], downR[2], look[2]}
        };
    }

    /** row-vector rays (N×3) in vehicle → camera: ray_cam = ray_veh · R */
    public void vehicleToCamera(double vx, double vy, double vz, double[] out) {
        double[][] r = rotationCameraToVehicle();
        out[0] = vx * r[0][0] + vy * r[1][0] + vz * r[2][0];
        out[1] = vx * r[0][1] + vy * r[1][1] + vz * r[2][1];
        out[2] = vx * r[0][2] + vy * r[1][2] + vz * r[2][2];
    }

    static double[] cross(double[] a, double[] b) {
        return new double[]{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }

    static double norm(double[] a) {
        return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }

    static void scaleInPlace(double[] a, double s) {
        a[0] *= s;
        a[1] *= s;
        a[2] *= s;
    }
}
