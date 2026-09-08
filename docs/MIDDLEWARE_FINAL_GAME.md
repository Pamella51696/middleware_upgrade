# Connect `middleware_upgrade` → `middleware_final_game`

This repo (`middleware_upgrade`) is the **camera / stitch middleware**.
[middleware_final_game](https://github.com/Pamella51696/middleware_final_game) is the **game** repo. It currently holds the old single-file `VideoStreamingServer.java`.

Do **not** run both stitchers. Run **this** server; point the game at its HTTP URLs.

## Same API as the old game server

| URL | What the game uses |
|-----|-------------------|
| `http://<jetson-or-pc>:9090/play` | Browser player (HTML + MJPEG) |
| `http://<jetson-or-pc>:9090/stitch` | Live **MJPEG** 270° panorama |
| `http://<jetson-or-pc>:9090/` | Debug mosaic + sliders |
| `GET/POST /api/sliders` | Tune FOV / blend (CORS enabled for a separate game origin) |

CLI is unchanged from the old class:

```bash
java -cp "$OPENCV_JAR:out" -Djava.library.path="$OPENCV_NATIVE" \
  fisheye270.VideoStreamingServer 9090 front.mov left.mov right.mov rear.mov
```

Or:

```bash
mvn -q exec:java -Dexec.args="serve 9090 front.mov left.mov right.mov rear.mov"
```

Default videos still come from `config/cameras.yaml` if you omit the four paths.

## What the game should do

1. Start middleware on the Jetson (or PC) as above.
2. In the game, set the stream base URL (example):

   ```text
   MIDDLEWARE_URL=http://192.168.1.50:9090
   ```

3. Show the panorama with:

   ```html
   <img src="http://192.168.1.50:9090/stitch" alt="surround">
   ```

   or open `/play` in a WebView.

4. Feature detection / AI (YOLO, Cosmos Reason 2 2B, etc.) should read **frames from `/stitch`**, not from a second copy of `VideoStreamingServer.java`.

## Git remotes (this clone)

```bash
git remote add final-game https://github.com/Pamella51696/middleware_final_game.git
git fetch final-game
```

`final-game` is the game repo. `origin` stays `middleware_upgrade`.

## Jetson

See `docs/JETSON.md`. Serve on port **9090**, then from the laptop/game machine use the Jetson LAN IP, not `localhost`.
