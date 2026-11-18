"""Live demo: the camera watches the board; press SPACE to read the position and get Stockfish's
best move for both sides.

Pipeline step: presentation. Ties every stage together, no new CV:
  SPACE -> latest frame -> recognize.py (warp, 64 crops, ResNet18, king rule) -> FEN
        -> analyze.py (Stockfish, both sides) -> side panel
Window: left = live warped board (exactly what the model sees), right = analysis panel.

Correction (human in the loop): click a square on the panel's board to fix it, and Stockfish re-runs.
  left-click  = the model's next-best guess for that square (mistakes are usually near-misses, e.g. king vs bishop)
  right-click = cycle through all 13 options
  s           = save this frame + the corrected FEN as a new training game in data/raw/live_<date>/
                (same format as capture.py). Retraining on these fixes the model's blind spots over time,
                e.g. kings on squares it never saw in training. Disabled with --images: re-saving test
                frames as training data would leak the test set.

Decisions (2026-11-17, PROJECT_LOG.md): OpenCV window instead of a web dashboard; keyboard trigger;
an impossible board is still shown, with a warning asking for another press; correction by next-best guess, saved.

Usage (from chess_vision/):
  python scripts/demo.py                                live camera (Iriun at 4K, after calibrate.py)
  python scripts/demo.py --images data/raw/test02       offline: each SPACE analyzes the next saved frame
Keys: SPACE = analyze   s = save correction   q / Esc = quit   mouse = correct squares (see above)
"""
import argparse
import csv
import datetime
import textwrap
from pathlib import Path

import chess
import chess.engine
import cv2
import numpy as np

import board as geo
from analyze import DEPTH, analyze, find_stockfish
from camera import ROOT, open_camera
from recognize import fix_kings, load_model, square_logits, to_fen

WINDOW = "Chess Vision"
PREVIEW = 860             # warped 1344 px board shown at 860x860
PANEL_W = 560
CELL = 52                 # panel board square size (8 x 52 = 416 px)
BOARD_X, BOARD_Y = 72, 70
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, RED, GREEN, YELLOW = (255, 255, 255), (170, 170, 170), (80, 80, 255), (120, 220, 120), (0, 220, 255)
LIGHT_SQ, DARK_SQ = (205, 228, 232), (95, 145, 100)   # BGR, roughly the green tournament board
LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}
JPEG_QUALITY = 95         # same as capture.py
# right-click order: empty, then White's pieces, then Black's
CYCLE = ["empty"] + [f"{c}_{p}" for c in ("white", "black") for p in ("pawn", "knight", "bishop", "rook", "queen", "king")]


def text(img, s, y, color=WHITE, scale=0.6, x=20, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_board(img, placement, corrected=()):
    """Recognized position as a diagram, White at the bottom. Pieces are discs with a letter (OpenCV fonts have no chess symbols).
    Squares fixed by hand get a yellow frame."""
    position = chess.BaseBoard(placement)
    for row in range(8):
        for col in range(8):
            x, y = BOARD_X + col * CELL, BOARD_Y + row * CELL
            cv2.rectangle(img, (x, y), (x + CELL, y + CELL), LIGHT_SQ if (row + col) % 2 == 0 else DARK_SQ, -1)
            piece = position.piece_at(chess.square(col, 7 - row))   # row 0 = rank 8
            if piece:
                fill, ink = ((240, 240, 240), (20, 20, 20)) if piece.color else ((25, 25, 25), (235, 235, 235))
                c = (x + CELL // 2, y + CELL // 2)
                cv2.circle(img, c, CELL // 2 - 5, fill, -1, cv2.LINE_AA)
                cv2.circle(img, c, CELL // 2 - 5, (128, 128, 128), 1, cv2.LINE_AA)
                text(img, LETTER[piece.piece_type], y + CELL // 2 + 8, ink, 0.7, x + CELL // 2 - 9, 2)
    for i in corrected:
        x, y = BOARD_X + (i % 8) * CELL, BOARD_Y + (i // 8) * CELL
        cv2.rectangle(img, (x + 1, y + 1), (x + CELL - 2, y + CELL - 2), YELLOW, 3)
    for i in range(8):   # file letters and rank numbers
        text(img, "abcdefgh"[i], BOARD_Y + 8 * CELL + 20, GREY, 0.5, BOARD_X + i * CELL + CELL // 2 - 5)
        text(img, str(8 - i), BOARD_Y + i * CELL + CELL // 2 + 6, GREY, 0.5, BOARD_X - 20)


def draw_panel(result=None, status=None):
    """Right-hand panel. result = dict from recognize + refresh, status = a one-line message (e.g. 'analyzing...')."""
    panel = np.full((PREVIEW, PANEL_W, 3), 32, np.uint8)
    text(panel, "CHESS VISION", 40, WHITE, 0.8, thick=2)
    y = BOARD_Y + 8 * CELL + 60
    if result:
        draw_board(panel, result["placement"], result["corrected"])
        for side, move in result["moves"].items():
            for i, line in enumerate(textwrap.wrap(f"{side} to move: {move}", 42)):
                text(panel, line, y, GREEN if i == 0 and not move.startswith("impossible") else GREY, 0.65)
                y += 28
        y += 6
        for note in result["notes"]:
            text(panel, note, y, GREY, 0.5)
            y += 22
        if result["impossible"]:
            text(panel, "IMPOSSIBLE BOARD: check pieces / hands,", y + 10, RED, 0.6, thick=2)
            text(panel, "then press SPACE again (or click to fix)", y + 36, RED, 0.6, thick=2)
            y += 56
        for i, line in enumerate(textwrap.wrap(result["placement"], 48)):
            text(panel, line, PREVIEW - 90 + i * 18, GREY, 0.45)
    if status:
        text(panel, status, y + 20, YELLOW, 0.6, thick=2)
    text(panel, "click: next guess   right-click: cycle", PREVIEW - 44, GREY, 0.5)
    text(panel, "SPACE analyze   s save   q quit", PREVIEW - 20, GREY, 0.55)
    return panel


def recognize(frame, model, classes, H, boxes):
    """Model + king rule on one frame. Keeps the logits, so corrections can step through next-best guesses."""
    names, logits = square_logits(frame, model, H, boxes)
    labels, changes = fix_kings(logits, classes)
    return {"frame": frame.copy(), "names": names, "logits": logits, "labels": labels, "corrected": set(),
            "notes": [f"king rule: {names[i]} {old} -> {new}" for i, old, new in changes], "saved": False}


def refresh(result, engine, depth):
    """Labels -> FEN -> Stockfish. Runs after recognizing and after every correction.
    Impossible = no side can legally be to move (Stockfish skipped both)."""
    result["placement"] = to_fen(result["names"], result["labels"])
    result["moves"] = analyze(engine, result["placement"], depth)
    result["impossible"] = all(m.startswith("impossible") for m in result["moves"].values())


def square_at(x, y):
    """Window pixel -> square index 0..63 in FEN order (a8 = 0, h1 = 63), or None if outside the panel board."""
    bx, by = x - PREVIEW - BOARD_X, y - BOARD_Y
    if 0 <= bx < 8 * CELL and 0 <= by < 8 * CELL:
        return (by // CELL) * 8 + bx // CELL
    return None


def correct(result, i, classes, next_guess):
    """Change square i: next_guess=True -> the model's next-best class for it, else the next class in CYCLE."""
    old = result["labels"][i]
    order = [classes[j] for j in np.argsort(-result["logits"][i])] if next_guess else CYCLE
    new = order[(order.index(old) + 1) % len(order)]
    result["labels"][i] = new
    result["corrected"].add(i)
    result["saved"] = False   # the board changed, so it can be saved again
    result["notes"] = (result["notes"] + [f"corrected {result['names'][i]}: {old} -> {new}"])[-6:]   # panel fits ~6 notes


def live_folder():
    """Today's live_<date> folder, holding a copy of the current calibration. If the camera was
    recalibrated since that folder started, use live_<date>_2, _3 ... so one folder never mixes calibrations."""
    current = geo.CALIB_PATH.read_bytes()
    base = f"live_{datetime.date.today().isoformat()}"
    n = 1
    while True:
        folder = ROOT / "data" / "raw" / (base if n == 1 else f"{base}_{n}")
        calib = folder / "calibration.json"
        if not calib.exists():
            folder.mkdir(parents=True, exist_ok=True)
            calib.write_bytes(current)
            return folder
        if calib.read_bytes() == current:
            return folder
        n += 1


def save_correction(result):
    """Append the frame + corrected FEN to data/raw/live_<date>/, the layout capture.py writes,
    so build_dataset.py turns it into training crops. Returns the saved path for the panel."""
    folder = live_folder()
    labels_csv = folder / "labels.csv"
    new_file = not labels_csv.exists()
    name = f"{len(list(folder.glob('*.jpg'))) + 1:04d}.jpg"
    if not cv2.imwrite(str(folder / name), result["frame"], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise RuntimeError(f"could not write {folder / name}")
    with open(labels_csv, "a", newline="") as f:
        out = csv.writer(f)
        if new_file:
            out.writerow(["file", "fen", "moves"])
        out.writerow([name, result["placement"] + " w - - 0 1", ""])   # turn unknown; only the placement is the label
    result["saved"] = True
    return f"{folder.name}/{name}"


def main():
    ap = argparse.ArgumentParser(description="Live board recognition + Stockfish in an OpenCV window.")
    ap.add_argument("--images", type=Path, help="folder of saved 4K frames to step through instead of the camera")
    ap.add_argument("--depth", type=int, default=DEPTH)
    args = ap.parse_args()

    model, classes = load_model()
    H, foot, lean = geo.load_calibration()
    boxes = geo.crop_boxes(foot, lean)
    files = sorted(args.images.glob("*.jpg")) if args.images else []
    if args.images and not files:
        raise SystemExit(f"no .jpg frames in {args.images}")
    cap = None if files else open_camera()
    index, frame = -1, cv2.imread(str(files[0])) if files else None
    result, status = None, "press SPACE to analyze"
    clicks = []
    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, lambda event, x, y, *_: clicks.append((event, x, y))
                         if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) else None)

    with chess.engine.SimpleEngine.popen_uci(find_stockfish()) as engine:
        try:
            while True:
                if cap:
                    ok, frame = cap.read()   # keep reading so the preview is live and SPACE uses the newest frame
                    if not ok:
                        raise RuntimeError("camera returned no frame")
                preview = cv2.resize(geo.warp(frame, H), (PREVIEW, PREVIEW), interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW, np.hstack([preview, draw_panel(result, status)]))
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if files:   # offline: step to the next saved frame
                        index = (index + 1) % len(files)
                        frame = cv2.imread(str(files[index]))
                    cv2.imshow(WINDOW, np.hstack([preview, draw_panel(status="analyzing...")]))
                    cv2.waitKey(1)   # let the window repaint before the ~5 s of work
                    result, status = recognize(frame, model, classes, H, boxes), None
                    if files:
                        result["notes"].insert(0, f"frame {files[index].parent.name}/{files[index].name}")
                    refresh(result, engine, args.depth)
                if key == ord("s") and result:
                    if files:
                        status = "saving is off with --images (test leak)"
                    elif result["saved"]:
                        status = "already saved"
                    else:
                        status = "saved " + save_correction(result)

                while clicks and result:
                    event, x, y = clicks.pop(0)
                    i = square_at(x, y)
                    if i is not None:
                        correct(result, i, classes, next_guess=(event == cv2.EVENT_LBUTTONDOWN))
                        refresh(result, engine, args.depth)
                        status = None
                clicks.clear()   # clicks before the first analysis do nothing
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
