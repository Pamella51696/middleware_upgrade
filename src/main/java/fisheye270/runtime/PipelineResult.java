package fisheye270.runtime;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class PipelineResult {
    public int index;
    public final Map<String, Mat> raw = new LinkedHashMap<>();
    public final Map<String, Mat> undistorted = new LinkedHashMap<>();
    public final Map<String, Mat> projected = new LinkedHashMap<>();
    public final Map<String, Mat> valid = new LinkedHashMap<>();
    public Mat panorama;
    public Mat seams;
    public final List<String> missing = new ArrayList<>();
    public double totalMs;
}
