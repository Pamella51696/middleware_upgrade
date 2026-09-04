"""Flask debug UI: eight views + live sliders. Maps rebuild only when sliders change."""

from __future__ import annotations

from typing import Any, Dict

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from ..config_loader import camera_cfg
from ..runtime.processor import Processor
from .mosaic import mosaic_from_result

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/>
<title>fisheye_270 debug</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: ui-sans-serif, system-ui; background:#0b0d12; color:#e8edf4; }
  header { padding:12px 18px; border-bottom:1px solid #2a3140; display:flex; gap:16px; align-items:center; }
  h1 { font-size:16px; font-weight:600; margin:0; letter-spacing:.04em; }
  .tabs button, select { background:#1a2030; color:#e8edf4; border:1px solid #3a4558; padding:6px 10px; border-radius:6px; cursor:pointer; }
  .tabs button.active { background:#2d6cdf; border-color:#2d6cdf; }
  main { display:grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 54px); }
  aside { padding:14px; border-right:1px solid #2a3140; overflow:auto; }
  label { display:flex; justify-content:space-between; font-size:12px; margin-top:10px; color:#9aa6b8; }
  input[type=range] { width:100%; }
  .stage { padding:12px; display:flex; align-items:center; justify-content:center; background:#08090c; }
  .stage img { max-width:100%; max-height: calc(100vh - 80px); border-radius:8px; }
  .meta { font-size:12px; color:#8b97a8; }
</style></head>
<body>
<header>
  <h1>FISHEYE 270° — DEBUG</h1>
  <div class="tabs" id="tabs"></div>
  <span class="meta" id="meta"></span>
</header>
<main>
  <aside id="sliders"></aside>
  <div class="stage"><img id="view" alt="view"/></div>
</main>
<script>
const views = [
  ["raw","1 Raw"],["undistorted","2 Undistort"],["projected","3 Projected"],
  ["overlap","4 Overlap"],["compare_proj","4b Proj compare"],
  ["stitch","7 Stitch"],["seams","8 Seams"]
];
const sliders = [
  ["fov_horizontal_deg", 120, 220, 1],
  ["fx", 80, 800, 1], ["fy", 80, 800, 1],
  ["cx", 0, 1280, 1], ["cy", 0, 800, 1],
  ["k1", -0.5, 0.5, 0.001], ["k2", -0.3, 0.3, 0.001],
  ["k3", -0.2, 0.2, 0.001], ["k4", -0.2, 0.2, 0.001],
  ["yaw_deg", -180, 180, 0.1], ["pitch_deg", -30, 30, 0.1], ["roll_deg", -20, 20, 0.1],
  ["dx", -200, 200, 0.5], ["dy", -120, 120, 0.5],
  ["scale", 0.8, 1.2, 0.001], ["blend_width_px", 4, 200, 1],
  ["output_fov", 180, 360, 1]
];
let view = "raw";
let cam = "FRONT";
function tabs(){
  const el = document.getElementById("tabs");
  el.innerHTML = views.map(([id,l]) => `<button data-v="${id}" class="${id===view?'active':''}">${l}</button>`).join(" ")
    + ` <select id="cam">${["FRONT","LEFT","RIGHT","REAR"].map(c=>`<option ${c===cam?"selected":""}>${c}</option>`).join("")}</select>`;
  el.querySelectorAll("button").forEach(b => b.onclick = () => { view=b.dataset.v; tabs(); stream(); });
  document.getElementById("cam").onchange = e => { cam=e.target.value; loadSliders(); };
}
function stream(){ document.getElementById("view").src = "/stream/" + view + "?t=" + Date.now(); }
async function loadSliders(){
  const cfg = await (await fetch("/api/sliders?camera="+cam)).json();
  const box = document.getElementById("sliders");
  box.innerHTML = sliders.map(([k,mn,mx,st]) => {
    const v = cfg[k] ?? mn;
    return `<label>${k} <span id="v_${k}">${v}</span></label>
      <input type="range" min="${mn}" max="${mx}" step="${st}" value="${v}" data-k="${k}"/>`;
  }).join("");
  box.querySelectorAll("input").forEach(inp => inp.oninput = () => {
    document.getElementById("v_"+inp.dataset.k).textContent = inp.value;
    fetch("/api/sliders", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({camera: cam, key: inp.dataset.k, value: parseFloat(inp.value)})});
  });
}
tabs(); loadSliders(); stream();
setInterval(stream, 700);
setInterval(async () => {
  const s = await (await fetch("/api/stats")).json();
  document.getElementById("meta").textContent =
    `fps ${s.fps.toFixed(1)}  lat ${s.latency_ms.toFixed(0)}ms  cpu ${s.cpu_percent}  rss ${s.rss_mb.toFixed(0)}MB`;
}, 1000);
</script></body></html>
"""


def create_app(proc: Processor, next_bundle, lock=None) -> Flask:
    app = Flask(__name__)
    state: Dict[str, Any] = {"result": None}

    def current():
        bundle = next_bundle()
        if bundle is not None:
            state["result"] = proc.process_bundle(bundle)
        return state["result"]

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.get("/stream/<view>")
    def stream(view: str):
        def gen():
            while True:
                result = current()
                if result is None:
                    break
                img = mosaic_from_result(proc, result, view)
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/stats")
    def stats():
        return jsonify(proc.perf.snapshot())

    @app.get("/api/sliders")
    def get_sliders():
        cam_id = request.args.get("camera", "FRONT")
        cam = camera_cfg(proc.cfg, cam_id)
        model = proc.models.get(cam_id)
        intra = (model.intrinsics if model else None)
        return jsonify({
            "fov_horizontal_deg": cam.get("fov_horizontal_deg") or 190,
            "fx": intra.fx if intra else 0,
            "fy": intra.fy if intra else 0,
            "cx": intra.cx if intra else 0,
            "cy": intra.cy if intra else 0,
            "k1": (cam.get("distortion") or {}).get("k1") or 0,
            "k2": (cam.get("distortion") or {}).get("k2") or 0,
            "k3": (cam.get("distortion") or {}).get("k3") or 0,
            "k4": (cam.get("distortion") or {}).get("k4") or 0,
            "yaw_deg": (cam.get("pose") or {}).get("yaw_deg") or 0,
            "pitch_deg": (cam.get("pose") or {}).get("pitch_deg") or 0,
            "roll_deg": (cam.get("pose") or {}).get("roll_deg") or 0,
            "dx": (cam.get("alignment") or {}).get("dx") or 0,
            "dy": (cam.get("alignment") or {}).get("dy") or 0,
            "scale": (cam.get("alignment") or {}).get("scale") or 1,
            "blend_width_px": proc.blend_width,
            "output_fov": proc.spec.fov_h_deg,
        })

    @app.post("/api/sliders")
    def set_sliders():
        data = request.get_json(force=True)
        cam_id = data.get("camera", "FRONT")
        key = data["key"]
        val = float(data["value"])
        cam = camera_cfg(proc.cfg, cam_id)
        if key == "fov_horizontal_deg":
            cam["fov_horizontal_deg"] = val
        elif key in ("fx", "fy", "cx", "cy"):
            cam.setdefault("intrinsics", {})[key] = val
            cam["intrinsics"]["source"] = "estimated"
        elif key in ("k1", "k2", "k3", "k4"):
            cam.setdefault("distortion", {})[key] = val
        elif key in ("yaw_deg", "pitch_deg", "roll_deg"):
            cam.setdefault("pose", {})[key] = val
        elif key in ("dx", "dy", "scale"):
            cam.setdefault("alignment", {})[key] = val
        elif key == "blend_width_px":
            proc.blend_width = int(val)
        elif key == "output_fov":
            proc.spec.fov_h_deg = val
        proc.models.pop(cam_id, None)
        proc.undistort_cache.pop(cam_id, None)
        return jsonify({"ok": True})

    return app
