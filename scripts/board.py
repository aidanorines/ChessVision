"""Board geometry: warp a camera frame to a flat top-down board, then cut the 64 square crops.

Pipeline step: preprocessing, run on EVERY frame (capture, dataset building, live recognition).
  frame (4K, perspective) --warp--> board (1344x1344, top-down) --crops--> 64 x (224x224) images

Warped layout, like a printed chess diagram: White at the bottom, a-file on the left.
  row 0 = rank 8, col 0 = file a. Each square is 112 px, with a 2-square (224 px) pad around the board.

Lean-aware crops: the warp is exact only for the board surface. A piece's top sits above it, so it
gets smeared straight AWAY from the "foot" (the point on the table under the camera), by an amount
that grows with distance from the foot:   top = base + lean * (base - foot)
(lean = h / (H - h) for a piece of height h and a camera at height H). calibrate.py measures foot and
lean once from the two kings' tops. Each crop then holds the piece's base plus the spot where the tallest
piece's top would lean to, and is resized to 224x224 (ResNet18's input size).

Run directly for a self-check: grabs a frame and saves two check images to outputs/.
"""
import json

import cv2
import numpy as np

from camera import ROOT, WIDTH, HEIGHT

SQ = 112                  # pixels per square in the warped board
PAD = 2 * SQ              # 224: room around the board for far pieces' tops, which lean up to ~2 squares out
SIZE = 8 * SQ + 2 * PAD   # 1344: full warped image width and height
OUT = 224                 # every crop is resized to ResNet18's input size
BASE_R = 0.6 * SQ         # room kept around a square's center for the piece base (covers sloppy placement)
FILES = "abcdefgh"
CALIB_PATH = ROOT / "calibration.json"


def square_name(row, col):
    return f"{FILES[col]}{8 - row}"   # row 0 is rank 8, col 0 is file a


def square_center(row, col):
    return np.array([PAD + (col + 0.5) * SQ, PAD + (row + 0.5) * SQ])


def homography(corners):
    """Solve the 3x3 matrix H that maps the photo onto the flat board, from the 4 clicked corners."""
    src = np.float32([corners["a1"], corners["h1"], corners["h8"], corners["a8"]])   # where the corners ARE
    far = PAD + 8 * SQ   # the board's far edge in the warped image
    dst = np.float32([[PAD, far], [far, far], [far, PAD], [PAD, PAD]])   # where they SHOULD BE (a1 bottom-left ...)
    return cv2.getPerspectiveTransform(src, dst)   # 4 point pairs = 8 equations = the 8 unknowns of H


def warp(frame, H):
    """Straighten one camera frame into the top-down board."""
    if frame.shape[:2] != (HEIGHT, WIDTH):
        raise RuntimeError(f"frame is {frame.shape[1]}x{frame.shape[0]}, expected {WIDTH}x{HEIGHT}")
    return cv2.warpPerspective(frame, H, (SIZE, SIZE))


def solve_lean(bases, tops):
    """Fit top = base + lean * (base - foot) to (base, top) pairs; return (foot, lean).
    Rearranged, top - base = lean*base - lean*foot is linear in the unknowns (lean, lean*foot_x, lean*foot_y),
    so ordinary least squares solves it. Each pair gives 2 equations; 2 pairs (4 equations) cover 3 unknowns."""
    A, y = [], []
    for (bx, by), (tx, ty) in zip(bases, tops):
        A += [[bx, -1, 0], [by, 0, -1]]
        y += [tx - bx, ty - by]
    (k, k_fx, k_fy), *_ = np.linalg.lstsq(np.array(A, float), np.array(y, float), rcond=None)
    return np.array([k_fx / k, k_fy / k]), k


def load_calibration(path=CALIB_PATH):
    """Return (H, foot, lean) from calibration.json. foot is stored in square units so it survives PAD/SQ changes."""
    calib = json.loads(path.read_text())
    if calib["frame_size"] != [WIDTH, HEIGHT]:
        raise RuntimeError(f"calibration was made on {calib['frame_size']} frames, camera gives {WIDTH}x{HEIGHT}")
    foot = PAD + SQ * np.array(calib["foot_sq"])
    return homography(calib["corners"]), foot, calib["lean"]


def game_calibration_path(game_dir):
    """The calibration a game was shot with: data/raw/<game>/calibration.json, copied there at capture time.
    Each game keeps its own, so recalibrating (camera moved) never changes how older photos are cropped."""
    path = game_dir / "calibration.json"
    if not path.exists():
        raise RuntimeError(f"{path} is missing: every game folder needs the calibration it was shot with")
    return path


def crop_boxes(foot, lean):
    """One square box (name, x, y, side) per board square, in FEN order: a8, b8 ... h8, a7 ... h1.
    Each box covers the base (radius BASE_R around the square center) and the tallest piece's leaning top."""
    boxes = []
    for row in range(8):
        for col in range(8):
            c = square_center(row, col)
            top = c + lean * (c - foot)                    # where a king's top would land
            side = np.abs(top - c).max() + 2 * BASE_R      # square box: fits the lean plus the base on both ends
            x, y = np.round((c + top) / 2 - side / 2).astype(int)   # centered halfway between base and top
            s = int(round(side))
            if x < 0 or y < 0 or x + s > SIZE or y + s > SIZE:
                raise RuntimeError(f"crop for {square_name(row, col)} leaves the warped image: increase PAD")
            boxes.append((square_name(row, col), x, y, s))
    return boxes


def crops(board, boxes):
    """Cut every box and resize it to OUT x OUT. Returns [(square_name, crop), ...] in FEN order."""
    out = []
    for name, x, y, s in boxes:
        interp = cv2.INTER_AREA if s > OUT else cv2.INTER_LINEAR   # AREA for shrinking, LINEAR for enlarging
        out.append((name, cv2.resize(board[y:y + s, x:x + s], (OUT, OUT), interpolation=interp)))
    return out


def _self_test():
    """solve_lean must recover a known foot and lean from synthetic tops."""
    foot, lean = np.array([80.0, 670.0]), 0.2
    bases = [square_center(7, 4), square_center(0, 4), square_center(7, 7)]
    tops = [b + lean * (b - foot) for b in bases]
    f, k = solve_lean(bases, tops)
    assert np.allclose(f, foot) and np.isclose(k, lean), (f, k)


if __name__ == "__main__":
    from camera import open_camera, grab_frame

    _self_test()
    H, foot, lean = load_calibration()
    boxes = crop_boxes(foot, lean)
    cap = open_camera()
    board = warp(grab_frame(cap), H)
    cap.release()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Check 1: grid, the foot direction, and a few crop boxes. Do the boxes hold each piece from base to top?
    view = board.copy()
    for i in range(9):
        p = PAD + i * SQ
        cv2.line(view, (p, PAD), (p, PAD + 8 * SQ), (0, 0, 255), 1)
        cv2.line(view, (PAD, p), (PAD + 8 * SQ, p), (0, 0, 255), 1)
    show = {"a1": (255, 0, 0), "h1": (0, 255, 255), "e1": (255, 0, 255), "a8": (0, 255, 0), "h8": (0, 128, 255), "e8": (255, 255, 0)}
    for name, x, y, s in boxes:
        if name in show:
            cv2.rectangle(view, (x, y), (x + s, y + s), show[name], 3)
            cv2.putText(view, name, (x + 6, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, show[name], 2)
    cv2.imwrite(str(out_dir / "check_grid.jpg"), view)

    # Check 2: all 64 crops tiled 8x8 like the board. Is every piece inside its tile, base and top?
    tiles = []
    for name, crop in crops(board, boxes):
        t = crop.copy()
        cv2.rectangle(t, (0, 0), (OUT - 1, OUT - 1), (255, 255, 255), 1)
        cv2.putText(t, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
    cv2.imwrite(str(out_dir / "check_crops.jpg"), np.vstack([np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(8)]))
    print(f"foot (warped px) = {foot.round(1)}, lean = {lean:.3f}, crop sides {min(b[3] for b in boxes)}-{max(b[3] for b in boxes)} px")
    print("saved", out_dir / "check_grid.jpg", "and", out_dir / "check_crops.jpg")
