package fisheye270.calibration;

/** Status of every geometric quantity. Never silently invent OEM accuracy. */
public enum Provenance {
    KNOWN, ASSUMED, ESTIMATED, CALIBRATED, UNKNOWN
}
