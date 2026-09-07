param(
    [string]$OpenCvJar = $env:OPENCV_JAR,
    [string]$OpenCvNative = $env:OPENCV_NATIVE
)

# From the repo root (the folder with pom.xml and config\), compile all Java
# sources using only opencv-490.jar. Do not javac *.java inside fisheye270\ —
# that misses calibration\, projection\, and the rest of the packages.

$ErrorActionPreference = "Stop"
if (-not $OpenCvJar) {
    $OpenCvJar = Join-Path $env:USERPROFILE "Downloads\opencv\build\java\opencv-490.jar"
}
if (-not $OpenCvNative) {
    $OpenCvNative = Join-Path $env:USERPROFILE "Downloads\opencv\build\java\x64"
}
if (-not (Test-Path $OpenCvJar)) {
    throw "Set OPENCV_JAR to opencv-490.jar (not found: $OpenCvJar)"
}

New-Item -ItemType Directory -Force -Path out | Out-Null
Get-ChildItem -Recurse -Filter *.java -Path src\main\java |
    ForEach-Object { $_.FullName } |
    Set-Content -Encoding ascii out\sources.txt
& javac -encoding UTF-8 -cp $OpenCvJar -d out "@out\sources.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Compiled to .\out"
Write-Host "java -cp `"$OpenCvJar;out`" -Djava.library.path=`"$OpenCvNative`" fisheye270.VideoStreamingServer 9090 front.mov left.mov right.mov rear.mov"
