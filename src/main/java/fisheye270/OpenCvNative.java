package fisheye270;

import org.opencv.core.Core;

/**
 * Loads OpenCV natives for both official Windows builds ({@code opencv_java490.dll})
 * and Maven OpenPnP ({@code nu.pattern.OpenCV}). Reflection keeps official
 * {@code opencv-490.jar} compilable without extra classes.
 */
public final class OpenCvNative {
    private OpenCvNative() {}

    public static void load() {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            return;
        } catch (UnsatisfiedLinkError first) {
            try {
                Class<?> c = Class.forName("nu.pattern.OpenCV");
                c.getMethod("loadLocally").invoke(null);
            } catch (ReflectiveOperationException | UnsatisfiedLinkError second) {
                throw new UnsatisfiedLinkError(
                        "OpenCV native library not found (" + Core.NATIVE_LIBRARY_NAME + "). "
                                + "Add opencv/build/java/x64 to java.library.path so opencv_java490.dll loads. "
                                + first.getMessage());
            }
        }
    }
}
