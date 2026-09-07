package fisheye270;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import fisheye270.config.AppConfig;
import fisheye270.input.FrameSynchronizer;
import fisheye270.runtime.PipelineResult;
import fisheye270.runtime.Processor;
import fisheye270.visualization.Mosaic;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;

/** MJPEG debug UI + panorama player (replaces the old single-class stitcher). */
public final class DebugServer {
    private DebugServer() {}

    public static void start(AppConfig cfg, Processor proc, FrameSynchronizer sync, int port) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/", ex -> html(ex, INDEX));
        server.createContext("/play", ex -> html(ex, PLAY));
        server.createContext("/stitch", ex -> stream(ex, proc, sync, "stitch"));
        server.createContext("/stream", ex -> {
            String q = ex.getRequestURI().getQuery();
            String view = "raw";
            if (q != null) {
                for (String p : q.split("&")) {
                    if (p.startsWith("view=")) view = URLDecoder.decode(p.substring(5), StandardCharsets.UTF_8);
                }
            }
            stream(ex, proc, sync, view);
        });
        server.createContext("/api/sliders", ex -> sliders(ex, cfg, proc));
        server.setExecutor(Executors.newFixedThreadPool(6));
        server.start();
        System.out.println("Server  ->  http://localhost:" + port + "/play");
        System.out.println("Debug   ->  http://localhost:" + port + "/");
    }

    static void html(HttpExchange ex, String html) throws IOException {
        byte[] b = html.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "text/html; charset=UTF-8");
        ex.sendResponseHeaders(200, b.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(b);
        }
    }

    static void stream(HttpExchange ex, Processor proc, FrameSynchronizer sync, String view) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(405, -1);
            return;
        }
        ex.getResponseHeaders().set("Content-Type", "multipart/x-mixed-replace; boundary=frame");
        ex.sendResponseHeaders(200, 0);
        try (OutputStream out = ex.getResponseBody()) {
            while (true) {
                FrameSynchronizer.Bundle b = sync.next();
                if (b == null) break;
                PipelineResult r = proc.process(b);
                Mat img = Mosaic.view(proc, r, view);
                byte[] jpeg = encode(img);
                String header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + jpeg.length + "\r\n\r\n";
                out.write(header.getBytes(StandardCharsets.UTF_8));
                out.write(jpeg);
                out.write("\r\n".getBytes(StandardCharsets.UTF_8));
                out.flush();
            }
        } catch (IOException ignored) {
            // client disconnect
        }
    }

    static byte[] encode(Mat frame) {
        MatOfByte buf = new MatOfByte();
        Imgcodecs.imencode(".jpg", frame, buf, new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 85));
        return buf.toArray();
    }

    static void sliders(HttpExchange ex, AppConfig cfg, Processor proc) throws IOException {
        if ("GET".equalsIgnoreCase(ex.getRequestMethod())) {
            String cam = param(ex, "camera", "FRONT");
            AppConfig.Cam c = cfg.cameras.get(cam);
            String json = "{"
                    + "\"fov_horizontal_deg\":" + c.fovH + ","
                    + "\"k1\":" + c.k1 + ","
                    + "\"yaw_deg\":" + c.pose.yawDeg + ","
                    + "\"dx\":" + c.alignment.dx + ","
                    + "\"dy\":" + c.alignment.dy + ","
                    + "\"scale\":" + c.alignment.scale + ","
                    + "\"blend_width_px\":" + cfg.blendWidthPx + ","
                    + "\"output_fov\":" + cfg.panorama.fovHDeg
                    + "}";
            byte[] b = json.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().set("Content-Type", "application/json");
            ex.sendResponseHeaders(200, b.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(b); }
            return;
        }
        if ("POST".equalsIgnoreCase(ex.getRequestMethod())) {
            byte[] body = ex.getRequestBody().readAllBytes();
            String s = new String(body, StandardCharsets.UTF_8);
            String cam = jsonStr(s, "camera", "FRONT");
            String key = jsonStr(s, "key", "");
            double val = jsonNum(s, "value", 0);
            AppConfig.Cam c = cfg.cameras.get(cam);
            switch (key) {
                case "fov_horizontal_deg" -> c.fovH = val;
                case "k1" -> c.k1 = val;
                case "yaw_deg" -> c.pose.yawDeg = val;
                case "dx" -> c.alignment.dx = val;
                case "dy" -> c.alignment.dy = val;
                case "scale" -> c.alignment.scale = val;
                case "blend_width_px" -> cfg.blendWidthPx = (int) val;
                case "output_fov" -> cfg.panorama.fovHDeg = val;
                default -> { }
            }
            proc.invalidate(cam);
            byte[] ok = "{\"ok\":true}".getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().set("Content-Type", "application/json");
            ex.sendResponseHeaders(200, ok.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(ok); }
        }
    }

    static String param(HttpExchange ex, String name, String def) {
        String q = ex.getRequestURI().getQuery();
        if (q == null) return def;
        for (String p : q.split("&")) {
            if (p.startsWith(name + "=")) return URLDecoder.decode(p.substring(name.length() + 1), StandardCharsets.UTF_8);
        }
        return def;
    }

    static String jsonStr(String s, String key, String def) {
        String needle = "\"" + key + "\"";
        int i = s.indexOf(needle);
        if (i < 0) return def;
        int c = s.indexOf(':', i);
        int q1 = s.indexOf('"', c + 1);
        if (q1 < 0) return def;
        int q2 = s.indexOf('"', q1 + 1);
        return s.substring(q1 + 1, q2);
    }

    static double jsonNum(String s, String key, double def) {
        String needle = "\"" + key + "\"";
        int i = s.indexOf(needle);
        if (i < 0) return def;
        int c = s.indexOf(':', i);
        String rest = s.substring(c + 1).trim();
        StringBuilder n = new StringBuilder();
        for (int k = 0; k < rest.length(); k++) {
            char ch = rest.charAt(k);
            if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '.' ) n.append(ch);
            else if (n.length() > 0) break;
        }
        try { return Double.parseDouble(n.toString()); } catch (Exception e) { return def; }
    }

    static final String PLAY = """
            <!DOCTYPE html><html><head><meta charset="UTF-8"><title>270° Panorama</title>
            <style>html,body{margin:0;height:100%;background:#0a0a0f;color:#e0e0e0;font-family:sans-serif}
            .c{display:flex;flex-direction:column;height:100vh;padding:12px;box-sizing:border-box}
            h1{font-weight:300;color:#7ec8e3;text-align:center}
            img{width:100%;flex:1;object-fit:contain;background:#000;border-radius:8px}</style></head>
            <body><div class="c"><h1>270° Cylindrical Panorama (Java / OpenCV)</h1>
            <img src="/stitch" alt="panorama"></div></body></html>
            """;

    static final String INDEX = """
            <!DOCTYPE html><html><head><meta charset="UTF-8"><title>fisheye_270 debug</title>
            <style>
            body{margin:0;font-family:sans-serif;background:#0b0d12;color:#e8edf4}
            header{padding:12px 18px;border-bottom:1px solid #2a3140;display:flex;gap:12px;align-items:center}
            button,select{background:#1a2030;color:#e8edf4;border:1px solid #3a4558;padding:6px 10px;border-radius:6px}
            button.active{background:#2d6cdf;border-color:#2d6cdf}
            main{display:grid;grid-template-columns:260px 1fr;min-height:calc(100vh - 52px)}
            aside{padding:14px;border-right:1px solid #2a3140}
            label{display:flex;justify-content:space-between;font-size:12px;margin-top:10px;color:#9aa6b8}
            input[type=range]{width:100%}
            .stage{display:flex;align-items:center;justify-content:center;background:#08090c}
            .stage img{max-width:100%;max-height:calc(100vh - 80px)}
            </style></head><body>
            <header><strong>FISHEYE 270° — JAVA DEBUG</strong><div id="tabs"></div>
            <select id="cam"><option>FRONT</option><option>LEFT</option><option>RIGHT</option><option>REAR</option></select>
            </header>
            <main><aside id="sliders"></aside><div class="stage"><img id="view"></div></main>
            <script>
            const views=[["raw","1 Raw"],["undistorted","2 Undistort"],["projected","3 Projected"],
              ["overlap","4 Overlap"],["compare_proj","4b Proj"],["stitch","7 Stitch"],["seams","8 Seams"]];
            const sliders=[["fov_horizontal_deg",120,220,1],["k1",-0.5,0.5,0.001],["yaw_deg",-180,180,0.1],
              ["dx",-200,200,0.5],["dy",-120,120,0.5],["scale",0.8,1.2,0.001],
              ["blend_width_px",4,200,1],["output_fov",180,360,1]];
            let view="raw", cam="FRONT";
            function tabs(){const el=document.getElementById("tabs");
              el.innerHTML=views.map(([id,l])=>`<button data-v="${id}" class="${id===view?"active":""}">${l}</button>`).join(" ");
              el.querySelectorAll("button").forEach(b=>b.onclick=()=>{view=b.dataset.v;tabs();stream();});}
            function stream(){document.getElementById("view").src="/stream?view="+view+"&t="+Date.now();}
            async function load(){const cfg=await (await fetch("/api/sliders?camera="+cam)).json();
              const box=document.getElementById("sliders");
              box.innerHTML=sliders.map(([k,mn,mx,st])=>{const v=cfg[k]??mn;
                return `<label>${k}<span id="v_${k}">${v}</span></label>
                  <input type="range" min="${mn}" max="${mx}" step="${st}" value="${v}" data-k="${k}">`;}).join("");
              box.querySelectorAll("input").forEach(inp=>inp.oninput=()=>{
                document.getElementById("v_"+inp.dataset.k).textContent=inp.value;
                fetch("/api/sliders",{method:"POST",headers:{"Content-Type":"application/json"},
                  body:JSON.stringify({camera:cam,key:inp.dataset.k,value:parseFloat(inp.value)})});});}
            document.getElementById("cam").onchange=e=>{cam=e.target.value;load();};
            tabs();load();stream();setInterval(stream,800);
            </script></body></html>
            """;
}
