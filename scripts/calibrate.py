"""Calibration, run once (and again only if the camera or board moves). Two click phases:

  1. Corners: click the 4 corners of the 8x8 playing area (a1, h1, h8, a8) on the camera frame.
     board.py turns them into the homography that straightens every later frame.
  2. Lean: on the straightened board, click the very top of the white king (e1) and the black king (e8).
     Their tops show how far the tallest pieces lean and where the camera's foot point is,
     which board.py uses to size and place the lean-aware crops.
  The board must be in the STARTING POSITION, with the kings on e1 and e8.

Pipeline step: preprocessing setup. Output: calibration.json.

Usage (from chess_vision/):  python scripts/calibrate.py
  Left click = place the next point   Right click = undo   Enter = accept   Esc = quit without saving
"""
import json
from datetime import date

import cv2
import numpy as np

import board
from camera import WIDTH, HEIGHT, open_camera, grab_frame

CORNER_NAMES = ["a1", "h1", "h8", "a8"]   # click order; it fixes which way the warped board comes out
RED = (0, 0, 255)                         # OpenCV colors are BGR, not RGB


def click_points(image, names, scale, outline):
    """Show `image` at `scale` and collect one click per name, in order.
    Returns full-resolution (x, y) points, or None if the user pressed Esc / closed the window."""
    small = cv2.resize(image, None, fx=scale, fy=scale)
    points = []

    def on_mouse(event, x, y, flags, param):
        # OpenCV calls this on every mouse event; x, y are in the scaled display, so divide by scale
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(names):
            points.append((round(x / scale), round(y / scale)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = "calibrate"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = small.copy()   # redraw on a fresh copy each loop so undone points disappear
        shown = [(round(x * scale), round(y * scale)) for x, y in points]
        if outline and len(shown) >= 2:
            cv2.polylines(view, [np.array(shown)], isClosed=len(shown) == len(names), color=RED, thickness=2)
        for name, p in zip(names, shown):
            cv2.circle(view, p, 6, RED, -1)
            cv2.putText(view, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
        msg = f"click: {names[len(points)]}" if len(points) < len(names) else "Enter = accept   right click = undo"
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(20)
        if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # Esc, or window closed with X
            cv2.destroyAllWindows()
            return None
        if key == 13 and len(points) == len(names):   # Enter
            cv2.destroyAllWindows()
            return points


def main():
    cap = open_camera()
    frame = grab_frame(cap)
    cap.release()

    # Phase 1: corners on the raw 4K frame (shown at half size to fit the screen)
    clicked = click_points(frame, CORNER_NAMES, scale=0.5, outline=True)
    if clicked is None:
        print("quit, nothing saved")
        return
    corners = dict(zip(CORNER_NAMES, clicked))

    # Phase 2: king tops on the warped board (1344 px, shown at 3/4 size)
    warped = board.warp(frame, board.homography(corners))
    tops = click_points(warped, ["top of white king (e1)", "top of black king (e8)"], scale=0.75, outline=False)
    if tops is None:
        print("quit, nothing saved")
        return
    bases = [board.square_center(7, 4), board.square_center(0, 4)]   # e1 and e8: the warp places bases exactly
    foot, lean = board.solve_lean(bases, tops)
    if not 0.05 < lean < 0.6:   # a misclick gives a nonsense lean; a real camera here gives about 0.2
        print(f"lean = {lean:.3f} looks wrong (expected about 0.2). Nothing saved; run again.")
        return

    calib = {
        "frame_size": [WIDTH, HEIGHT],   # board.py checks frames match the size these points came from
        "corners": corners,
        "foot_sq": ((foot - board.PAD) / board.SQ).round(3).tolist(),   # in squares from the a8 corner
        "lean": round(float(lean), 4),
        "created": date.today().isoformat(),
    }
    board.CALIB_PATH.write_text(json.dumps(calib, indent=2))
    print("saved", board.CALIB_PATH)
    print(f"foot = {calib['foot_sq']} squares from the a8 corner (x right, y down), lean = {calib['lean']}")


if __name__ == "__main__":
    main()
