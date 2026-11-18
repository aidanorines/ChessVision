"""Live demo: the camera watches the board; press SPACE to read the position and get Stockfish's
best move for both sides.

Pipeline step: presentation. Ties every stage together, no new CV:
  SPACE -> latest frame -> recognize.py (warp, 64 crops, ResNet18, king rule) -> FEN
        -> analyze.py (Stockfish, both sides) -> side panel
Window: left = live warped board (exactly what the model sees), right = analysis panel.

Decisions (2026-11-17, PROJECT_LOG.md): OpenCV window instead of a web dashboard; keyboard trigger;
an impossible board is still shown, with a warning asking for another press.

Usage (from chess_vision/):
  python scripts/demo.py                                live camera (Iriun at 4K, after calibrate.py)
  python scripts/demo.py --images data/raw/test02       offline: each SPACE analyzes the next saved frame
Keys: SPACE = analyze   q / Esc = quit
"""
import argparse
import textwrap
from pathlib import Path

import chess
import chess.engine
import cv2
import numpy as np

import board as geo
from analyze import DEPTH, analyze, find_stockfish
from camera import open_camera
from recognize import fix_kings, load_model, square_logits, to_fen

WINDOW = "Chess Vision"
PREVIEW = 860             # warped 1344 px board shown at 860x860
PANEL_W = 560
CELL = 52                 # panel board square size (8 x 52 = 416 px)
BOARD_X, BOARD_Y = 72, 70
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, RED, GREEN = (255, 255, 255), (170, 170, 170), (80, 80, 255), (120, 220, 120)
LIGHT_SQ, DARK_SQ = (205, 228, 232), (95, 145, 100)   # BGR, roughly the green tournament board
LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}


def text(img, s, y, color=WHITE, scale=0.6, x=20, thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_board(img, placement):
    """Recognized position as a diagram, White at the bottom. Pieces are discs with a letter (OpenCV fonts have no chess symbols)."""
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
    for i in range(8):   # file letters and rank numbers
        text(img, "abcdefgh"[i], BOARD_Y + 8 * CELL + 20, GREY, 0.5, BOARD_X + i * CELL + CELL // 2 - 5)
        text(img, str(8 - i), BOARD_Y + i * CELL + CELL // 2 + 6, GREY, 0.5, BOARD_X - 20)


def draw_panel(result=None, status=None):
    """Right-hand panel. result = dict from recognize_and_analyze, status = a one-line message (e.g. 'analyzing...')."""
    panel = np.full((PREVIEW, PANEL_W, 3), 32, np.uint8)
    text(panel, "CHESS VISION", 40, WHITE, 0.8, thick=2)
    y = BOARD_Y + 8 * CELL + 60
    if result:
        draw_board(panel, result["placement"])
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
            text(panel, "then press SPACE again", y + 36, RED, 0.6, thick=2)
            y += 56
        for i, line in enumerate(textwrap.wrap(result["placement"], 48)):
            text(panel, line, PREVIEW - 70 + i * 18, GREY, 0.45)
    if status:
        text(panel, status, y + 20, WHITE, 0.7, thick=2)
    text(panel, "SPACE analyze    q quit", PREVIEW - 20, GREY, 0.55)
    return panel


def recognize_and_analyze(frame, model, classes, H, boxes, engine, depth):
    """One full pipeline pass. Impossible = no side can legally be to move (Stockfish skipped both)."""
    names, logits = square_logits(frame, model, H, boxes)
    labels, changes = fix_kings(logits, classes)
    placement = to_fen(names, labels)
    moves = analyze(engine, placement, depth)
    return {"placement": placement, "moves": moves,
            "notes": [f"king rule: {names[i]} {old} -> {new}" for i, old, new in changes],
            "impossible": all(m.startswith("impossible") for m in moves.values())}


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
    panel = draw_panel(status="press SPACE to analyze")

    with chess.engine.SimpleEngine.popen_uci(find_stockfish()) as engine:
        try:
            while True:
                if cap:
                    ok, frame = cap.read()   # keep reading so the preview is live and SPACE uses the newest frame
                    if not ok:
                        raise RuntimeError("camera returned no frame")
                preview = cv2.resize(geo.warp(frame, H), (PREVIEW, PREVIEW), interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW, np.hstack([preview, panel]))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if files:   # offline: step to the next saved frame
                        index = (index + 1) % len(files)
                        frame = cv2.imread(str(files[index]))
                    cv2.imshow(WINDOW, np.hstack([preview, draw_panel(status="analyzing...")]))
                    cv2.waitKey(1)   # let the window repaint before the ~5 s of work
                    result = recognize_and_analyze(frame, model, classes, H, boxes, engine, args.depth)
                    if files:
                        result["notes"].insert(0, f"frame {files[index].parent.name}/{files[index].name}")
                    panel = draw_panel(result)
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
