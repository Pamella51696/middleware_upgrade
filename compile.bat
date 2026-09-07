@echo off
setlocal
REM Compile every .java file with ONLY the official OpenCV JAR (no SnakeYAML).
REM Run this from the folder that contains pom.xml / config / src.

if "%OPENCV_JAR%"=="" set "OPENCV_JAR=C:\Users\%USERNAME%\Downloads\opencv\build\java\opencv-490.jar"
if "%OPENCV_NATIVE%"=="" set "OPENCV_NATIVE=C:\Users\%USERNAME%\Downloads\opencv\build\java\x64"

if not exist "%OPENCV_JAR%" (
  echo Set OPENCV_JAR to your opencv-490.jar
  echo Example: set OPENCV_JAR=C:\Users\ps95973\Downloads\opencv\build\java\opencv-490.jar
  exit /b 1
)

if not exist out mkdir out
dir /s /b src\main\java\*.java > out\sources.txt
javac -encoding UTF-8 -cp "%OPENCV_JAR%" -d out @out\sources.txt
if errorlevel 1 exit /b 1

echo Compiled. Example run:
echo   java -cp "%OPENCV_JAR%;out" -Djava.library.path="%OPENCV_NATIVE%" fisheye270.VideoStreamingServer 9090 front.mov left.mov right.mov rear.mov
endlocal
