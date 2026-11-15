"""Camera access for the chess vision pipeline.

Pipeline step: input. Every other script (calibration, capture, live recognition) gets its frames
from here, so the camera settings live in exactly one place.
Run directly for a self-check: saves one frame to chess_vision/test_frame.jpg.
"""
import ctypes
import sys
from pathlib import Path

import cv2

# Windows display scaling (150% on this laptop) blows OpenCV windows up 1.5x, past the screen edge.
# Declaring DPI awareness makes 1 image pixel = 1 screen pixel, so a 1920x1080 window fits the 2240x1400 screen.
if sys.platform == "win32":
    ctypes.windll.shcore.SetProcessDpiAwareness(2)

ROOT = Path(__file__).resolve().parent.parent   # chess_vision/ (this file is in chess_vision/scripts/)
CAMERA_INDEX = 2             # Iriun Webcam on this laptop (0 = built-in webcam, 1 = Camo)
WIDTH, HEIGHT = 3840, 2160   # 4K. Iriun's own "Video format" dropdown must also be set to 3840x2160


def open_camera():
    """Open the phone camera stream at 4K."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)   # Media Foundation: DirectShow gave black frames from virtual cameras
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)               # a request; the driver may give something else, so grab_frame checks
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(f"camera {CAMERA_INDEX} did not open: is Iriun running on both the phone and the laptop?")
    return cap


def grab_frame(cap, warmup=10):
    """Return one frame. Reads `warmup` frames and keeps the last, because the first frames
    after opening can be dark (exposure still adjusting) or stale (left in the buffer)."""
    for _ in range(warmup):
        ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera returned no frame")
    h, w = frame.shape[:2]
    # Calibration corners are stored in 4K pixel coordinates, so a 1080p frame would silently break everything.
    if (w, h) != (WIDTH, HEIGHT):
        raise RuntimeError(f"got {w}x{h}, expected {WIDTH}x{HEIGHT}: set Iriun's Video format to 3840x2160")
    return frame


if __name__ == "__main__":
    cap = open_camera()
    frame = grab_frame(cap)
    cap.release()
    out = ROOT / "test_frame.jpg"
    cv2.imwrite(str(out), frame)
    print("saved", out, "shape (height, width, channels):", frame.shape)
