import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class VideoStreamingServer {

    private static final int DEFAULT_PORT  = 9090;
    private static final int TARGET_HEIGHT = 360;
    private static final int TARGET_WIDTH  = 640;
    private static final int OVERLAP_PX    = 80;

    // =========================================================================
    public static void main(String[] args) throws IOException {

        Path frontVideo = Paths.get("rear_1.mov");
        Path rearVideo  = Paths.get("left_1.mov");
        Path sideVideo  = Paths.get("right_1.mov");
        Path backVideo  = Paths.get("front_1.mov");

        if (args.length >= 5) {
            frontVideo = Paths.get(args[1]);
            rearVideo  = Paths.get(args[2]);
            sideVideo  = Paths.get(args[3]);
            backVideo  = Paths.get(args[4]);
        }

        Path[] videos = { frontVideo, rearVideo, sideVideo, backVideo };

        for (Path v : videos) {
            if (!Files.exists(v) || Files.isDirectory(v)) {
                System.err.println("Video file not found: " + v);
                return;
            }
        }

        int port = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_PORT;

        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("OpenCV native library not found: " + e.getMessage());
            return;
        }

        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);

        server.createContext("/stitch", new StitchHandler(videos));
        server.createContext("/play",   new PlayerPageHandler());

        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();

        System.out.println("Server started  ->  http://localhost:" + port + "/play");
        System.out.println("Feeds: front=" + frontVideo + " rear=" + rearVideo
                + " side=" + sideVideo + " back=" + backVideo);
    }

    // STITCH HANDLER  -  undistort each feed, then feather-blend panorama

    private static class StitchHandler implements HttpHandler {
      private final Path[] videoFiles;
      StitchHandler(Path[] f) { this.videoFiles = f; }
      private int resetCount = 0;

      @Override public void handle(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
          ex.sendResponseHeaders(405, -1); return;
        }

        VideoCapture[] caps = new VideoCapture[videoFiles.length];
        for (int i = 0; i < videoFiles.length; i++) {
          caps[i] = new VideoCapture(videoFiles[i].toString());
          if (!caps[i].isOpened()) {
            ex.sendResponseHeaders(500, -1); return;
          }
        }

        FisheyeUndistorter[] undistort = new FisheyeUndistorter[videoFiles.length];
        for (int i = 0; i < undistort.length; i++) {
          undistort[i] = new FisheyeUndistorter();
        }

        ex.getResponseHeaders().set("Content-Type", "multipart/x-mixed-replace; boundary=frame");
        ex.sendResponseHeaders(200, 0);

        try (OutputStream out = ex.getResponseBody()) {
          Mat[] frames  = new Mat[videoFiles.length];
          Mat[] ready   = new Mat[videoFiles.length];
          for (int i = 0; i < videoFiles.length; i++) {
            frames[i] = new Mat();
            ready[i]  = new Mat();
          }

          while (true) {
            for (int i = 0; i < caps.length; i++) {
              boolean frameRead = caps[i].read(frames[i]);
              if (!frameRead || frames[i].empty()) {
                caps[i].release();
                caps[i] = new VideoCapture(videoFiles[i].toString());

                if (!caps[i].isOpened()) {
                  System.err.println("Error: Could not reopen video " + i);
                  continue;
                }

                frameRead = caps[i].read(frames[i]);
                if (!frameRead || frames[i].empty()) {
                  System.err.println("Warning: Could not read frame from video " + i + " after reopening");
                  continue;
                }
                resetCount++;
                System.out.println("Video " + i + " restarted (total resets: " + resetCount + ")");
              }

              undistort[i].recalibrateAndFilter(frames[i], ready[i]);
            }

            boolean allReady = true;
            for (int i = 0; i < ready.length; i++) {
              if (ready[i].empty()
                  || ready[i].cols() != TARGET_WIDTH
                  || ready[i].rows() != TARGET_HEIGHT) {
                allReady = false;
                break;
              }
            }
            if (!allReady) {
              continue;
            }

            Mat panorama = featherStitch(ready);
            writeFrame(out, encodeJpeg(panorama));
            panorama.release();
          }
        }
        finally {
          for (VideoCapture c : caps) c.release();
        }

      } // handle()
    } // StitchHandler


    // PLAYER PAGE  -  stitched panorama only, full-viewport

    private static class PlayerPageHandler implements HttpHandler {

        @Override public void handle(HttpExchange ex) throws IOException {
            String html = "<!DOCTYPE html><html lang='en'><head>"
                + "<meta charset='UTF-8'>"
                + "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                + "<title>360° Panoramic View</title>"
                + "<style>"
                + "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }"
                + "html, body { height: 100%; background: #0a0a0f; color: #e0e0e0;"
                + "  font-family: 'Segoe UI', sans-serif; overflow: hidden; }"
                + ".container { display: flex; flex-direction: column;"
                + "  align-items: center; justify-content: center;"
                + "  height: 100vh; padding: 16px; gap: 12px; }"
                + "h1 { font-size: 1.4rem; font-weight: 300; letter-spacing: 2px;"
                + "  color: #7ec8e3; text-align: center; flex-shrink: 0; }"
                + ".pano-wrap { width: 100%; flex: 1; min-height: 0;"
                + "  border: 1px solid #2a2a3a; border-radius: 8px; overflow: hidden;"
                + "  display: flex; align-items: center; justify-content: center; }"
                + ".pano-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; }"
                + "</style>"
                + "</head><body>"
                + "<div class='container'>"
                + "  <h1>360° Panoramic Camera System</h1>"
                + "  <div class='pano-wrap'>"
                + "    <img src='/stitch' alt='360° stitched panorama'>"
                + "  </div>"
                + "</div>"
                + "</body></html>";

            byte[] bytes = html.getBytes("UTF-8");
            ex.getResponseHeaders().set("Content-Type", "text/html; charset=UTF-8");
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
        }
    }


    //  FISHEYE UNDISTORT LAYER
    //
    //  Size-adaptive recalibration: K is rebuilt from the incoming frame size,
    //  so a new camera/resolution does not require retuning this block.
    //  Optional knobs (FOV / balance / generic D) live only here.
    //  Output of recalibrateAndFilter is always TARGET_WIDTH x TARGET_HEIGHT.

    static final class FisheyeUndistorter {

        /** Horizontal field of view assumed for an equidistant fisheye. */
        private static final double FISHEYE_FOV_DEG = 180.0;

        /** 0 = crop to valid pixels, 1 = keep full FOV (may show black edges). */
        private static final double BALANCE = 0.0;

        /** >1 zooms in (crops more), <1 keeps more of the rectified frame. */
        private static final double FOV_SCALE = 1.0;

        /**
         * Generic Kannala-Brandt fisheye coefficients (k1..k4).
         * Leave at zeros for a pure equidistant model derived from FOV + size.
         * Only change this if a new lens family is strongly non-equidistant.
         */
        private static final double[] FISHEYE_D = { 0.0, 0.0, 0.0, 0.0 };

        private Mat map1;
        private Mat map2;
        private Mat undistorted;
        private Mat filtered;
        private int cachedSrcW = -1;
        private int cachedSrcH = -1;

        /**
         * Recalibrate (maps rebuilt only when source size changes), undistort,
         * lightly filter, and fit to 360x640 for the stitcher.
         */
        void recalibrateAndFilter(Mat src, Mat dst360x640) {
            if (src == null || src.empty()) {
                return;
            }

            ensureMaps(src.cols(), src.rows());

            if (undistorted == null) undistorted = new Mat();
            if (filtered == null)    filtered    = new Mat();

            Imgproc.remap(src, undistorted, map1, map2, Imgproc.INTER_LINEAR,
                    Imgproc.BORDER_CONSTANT, Scalar.all(0));

            Imgproc.GaussianBlur(undistorted, filtered, new Size(3, 3), 0.8);

            Size target = new Size(TARGET_WIDTH, TARGET_HEIGHT);
            if (filtered.cols() == TARGET_WIDTH && filtered.rows() == TARGET_HEIGHT) {
                filtered.copyTo(dst360x640);
            } else {
                Imgproc.resize(filtered, dst360x640, target, 0, 0, Imgproc.INTER_AREA);
            }
        }

        private void ensureMaps(int srcW, int srcH) {
            if (map1 != null && srcW == cachedSrcW && srcH == cachedSrcH) {
                return;
            }

            Size srcSize = new Size(srcW, srcH);
            Size dstSize = new Size(TARGET_WIDTH, TARGET_HEIGHT);

            Mat K = cameraMatrixFromSize(srcW, srcH);
            Mat D = distortionCoeffs();
            Mat R = Mat.eye(3, 3, CvType.CV_64FC1);
            Mat P = new Mat();

            Calib3d.fisheye_estimateNewCameraMatrixForUndistortRectify(
                    K, D, srcSize, R, P, BALANCE, dstSize, FOV_SCALE);

            if (map1 == null) map1 = new Mat();
            if (map2 == null) map2 = new Mat();

            Calib3d.fisheye_initUndistortRectifyMap(
                    K, D, R, P, dstSize, CvType.CV_16SC2, map1, map2);

            cachedSrcW = srcW;
            cachedSrcH = srcH;

            K.release();
            D.release();
            R.release();
            P.release();
        }

        /**
         * Equidistant fisheye K from image size only:
         *   r = f * theta,  (width/2) = f * (fov/2 in radians)
         * Principal point is the frame center. Resolution changes adapt automatically.
         */
        private static Mat cameraMatrixFromSize(int width, int height) {
            double halfFov = Math.toRadians(FISHEYE_FOV_DEG) / 2.0;
            double fx = (width  / 2.0) / halfFov;
            double fy = (height / 2.0) / halfFov;

            Mat K = Mat.eye(3, 3, CvType.CV_64FC1);
            K.put(0, 0, fx);
            K.put(1, 1, fy);
            K.put(0, 2, width  / 2.0);
            K.put(1, 2, height / 2.0);
            return K;
        }

        private static Mat distortionCoeffs() {
            Mat D = new Mat(4, 1, CvType.CV_64FC1);
            D.put(0, 0, FISHEYE_D[0]);
            D.put(1, 0, FISHEYE_D[1]);
            D.put(2, 0, FISHEYE_D[2]);
            D.put(3, 0, FISHEYE_D[3]);
            return D;
        }
    }


    //  CORE BLENDING  —  featherStitch

    static Mat featherStitch(Mat[] frames) {
        int N = frames.length;
        int H = TARGET_HEIGHT;
        int W = TARGET_WIDTH;

        int overlap = Math.min(OVERLAP_PX, W / 4);
        int panoW   = W + (N - 1) * (W - overlap);

        Mat accumColor  = Mat.zeros(H, panoW, CvType.CV_32FC3);
        Mat accumWeight = Mat.zeros(H, panoW, CvType.CV_32FC1);

        for (int i = 0; i < N; i++) {
            int xStart = i * (W - overlap);

            Mat weight = buildFeatherMask(H, W, overlap);

            Mat frameF = new Mat();
            frames[i].convertTo(frameF, CvType.CV_32FC3);

            Mat weight3 = new Mat();
            List<Mat> ch = new ArrayList<>();
            ch.add(weight); ch.add(weight); ch.add(weight);
            Core.merge(ch, weight3);

            Mat wFrame = new Mat();
            Core.multiply(frameF, weight3, wFrame);

            int xEnd    = Math.min(xStart + W, panoW);
            int wActual = xEnd - xStart;

            Mat colorRoi  = accumColor.submat(0, H, xStart, xEnd);
            Mat weightRoi = accumWeight.submat(0, H, xStart, xEnd);

            Mat wFrameCrop = wFrame.colRange(0, wActual);
            Mat weightCrop = weight.colRange(0, wActual);

            Core.add(colorRoi,  wFrameCrop, colorRoi);
            Core.add(weightRoi, weightCrop, weightRoi);

            colorRoi.release(); weightRoi.release();
            frameF.release(); weight.release(); weight3.release();
            wFrame.release();
        }

        Mat safeW = new Mat();
        Core.max(accumWeight, new Scalar(1e-6), safeW);

        Mat safeW3 = new Mat();
        List<Mat> wch = new ArrayList<>();
        wch.add(safeW); wch.add(safeW); wch.add(safeW);
        Core.merge(wch, safeW3);

        Mat blended = new Mat();
        Core.divide(accumColor, safeW3, blended);

        Mat result = new Mat();
        blended.convertTo(result, CvType.CV_8UC3);

        accumColor.release(); accumWeight.release();
        safeW.release(); safeW3.release(); blended.release();

        return result;
    }

    static Mat buildFeatherMask(int H, int W, int overlap) {
        Mat mask = new Mat(H, W, CvType.CV_32FC1, new Scalar(1.0));
        for (int x = 0; x < overlap; x++) {
            float alpha = (float) x / overlap;
            for (int y = 0; y < H; y++) {
                mask.put(y, x,          new float[]{ alpha });
                mask.put(y, W - 1 - x,  new float[]{ alpha });
            }
        }
        return mask;
    }


    // SHARED HELPERS

    static byte[] encodeJpeg(Mat frame) {
        MatOfByte buf    = new MatOfByte();
        MatOfInt  params = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 88);
        Imgcodecs.imencode(".jpg", frame, buf, params);
        return buf.toArray();
    }

    static void writeFrame(OutputStream out, byte[] jpeg) throws IOException {
        String header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                      + jpeg.length + "\r\n\r\n";
        out.write(header.getBytes("UTF-8"));
        out.write(jpeg);
        out.write("\r\n".getBytes("UTF-8"));
        out.flush();
    }
}
