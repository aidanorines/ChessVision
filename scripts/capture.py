"""Labeled data capture: the script decides each position, you copy it onto the board, the remote saves the photo.

Pipeline step: data collection. Output for each game:
  data/raw/<game>/0001.jpg, 0002.jpg ...   raw 4K frames (crops can be rebuilt from these any time)
  data/raw/<game>/labels.csv               file, fen, moves: the label for each photo
  data/raw/<game>/calibration.json         copy of the calibration this game was shot with

Two modes:
  random (train/val): from the starting position, play 1-3 random legal moves per photo. Random play
                      scatters the pieces, so every square sees many different neighbours.
  --pgn  (test):      replay a real game, one full move (2 plies) per photo.
The starting position itself is never photographed (it would be identical in every game).

Usage (from chess_vision/):
  python scripts/capture.py g01                         random game, 40 photos
  python scripts/capture.py test01 --pgn pgn/game1.pgn  real game
Keys in the preview window:  r = retake the last photo (fix the board, press the remote again)   q = quit
"""
import argparse
import csv
import random
import shutil
from itertools import islice

import chess
import chess.pgn
import cv2

import board as geo   # our board.py (warp + crops); "geo" so it isn't confused with a chess board
from camera import ROOT, open_camera, grab_frame
from remote import start_remote

SPARE_QUEENS = 1          # extra queens per colour in the physical set; promotions beyond that are skipped
PREVIEW_SCALE = 0.65      # 1344 px warped board -> about 870 px on screen
JPEG_QUALITY = 95         # near-lossless; a 4K frame is about 3 MB
YELLOW = (0, 255, 255)


def random_moves(pos, rng):
    """Pick 1-3 random legal moves from `pos` (without changing it). Promotions are only to a queen,
    and only while a physical queen is left for that colour."""
    trial = pos.copy()
    moves = []
    for _ in range(rng.randint(1, 3)):
        if trial.is_game_over():
            break
        queens = len(trial.pieces(chess.QUEEN, trial.turn))
        legal = [m for m in trial.legal_moves
                 if m.promotion is None or (m.promotion == chess.QUEEN and queens < 1 + SPARE_QUEENS)]
        if not legal:
            break
        m = rng.choice(legal)
        moves.append(m)
        trial.push(m)
    return moves


def make_stepper(args):
    """Return a function pos -> list of moves for the next photo."""
    if not args.pgn:
        rng = random.Random(args.game)   # seeded by the game name, so a game's moves can be reproduced
        return lambda pos: random_moves(pos, rng)
    with open(args.pgn) as f:
        mainline = iter(chess.pgn.read_game(f).mainline_moves())
    return lambda pos: list(islice(mainline, 2))   # next 2 plies (one full move); islice never over-reads


def describe(pos, m):
    """Instruction for one move. Must be called BEFORE the move is played (SAN needs the old position)."""
    text = f"{pos.san(m):8} {chess.square_name(m.from_square)} -> {chess.square_name(m.to_square)}"
    if pos.is_castling(m):
        text += "   (castling: move the rook too)"
    if pos.is_en_passant(m):
        captured = m.to_square - 8 if pos.turn == chess.WHITE else m.to_square + 8
        text += f"   (en passant: remove the pawn on {chess.square_name(captured)})"
    if m.promotion:
        text += "   (promotion: replace the pawn with a spare queen)"
    return text


def main():
    ap = argparse.ArgumentParser(description="Capture labeled raw frames for one game.")
    ap.add_argument("game", help="folder name under data/raw, e.g. g01 or test01")
    ap.add_argument("--pgn", help="replay this real game (test set) instead of random moves")
    ap.add_argument("--photos", type=int, default=40)
    args = ap.parse_args()

    out = ROOT / "data" / "raw" / args.game
    if out.exists():
        raise SystemExit(f"{out} already exists. Pick a new game name; nothing is overwritten.")

    H, _, _ = geo.load_calibration()
    cap = open_camera()
    grab_frame(cap)                 # warm up and confirm 4K before anything is written
    presses = start_remote()
    out.mkdir(parents=True)
    shutil.copy(geo.CALIB_PATH, out / "calibration.json")   # this game's calibration, used when building its crops
    next_moves = make_stepper(args)
    pos = chess.Board()
    steps = []   # one entry per step: (plies pushed, SAN list, squares to highlight). Last one = waiting for a photo.
    rows = []    # saved photos: (file, fen, moves)

    def new_step():
        """Apply the next moves to `pos`, print the instructions, and return False if the game is over."""
        moves = next_moves(pos)
        if not moves:
            return False
        sans, squares = [], []
        print(f"\nPhoto {len(rows) + 1}/{args.photos}. Make these moves, then press the remote:")
        for m in moves:
            print("   ", describe(pos, m))
            sans.append(pos.san(m))
            squares += [m.from_square, m.to_square]
            pos.push(m)
        print(pos)   # text board, White at the bottom like the preview: compare it with the real board
        steps.append((len(moves), sans, squares))
        return True

    def write_labels():
        with open(out / "labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "fen", "moves"])
            w.writerows(rows)

    new_step()
    while not presses.empty():   # ignore presses from before the window opened
        presses.get()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            view = cv2.resize(geo.warp(frame, H), None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            for sq in steps[-1][2]:   # highlight the squares that change in this step
                row, col = 7 - chess.square_rank(sq), chess.square_file(sq)
                x, y = geo.PAD + col * geo.SQ, geo.PAD + row * geo.SQ
                p0 = (round(x * PREVIEW_SCALE), round(y * PREVIEW_SCALE))
                p1 = (round((x + geo.SQ) * PREVIEW_SCALE), round((y + geo.SQ) * PREVIEW_SCALE))
                cv2.rectangle(view, p0, p1, YELLOW, 3)
            msg = f"photo {len(rows) + 1}/{args.photos}: {' '.join(steps[-1][1])}   [remote] save  [r] retake  [q] quit"
            cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
            cv2.imshow("capture", view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r") and rows:
                # Throw away the last photo and go back to waiting for it: undo the step shown now,
                # so the step that photo belonged to becomes the current one again.
                for _ in range(steps.pop()[0]):
                    pos.pop()
                (out / rows.pop()[0]).unlink()
                write_labels()
                print(f"\nRetake photo {len(rows) + 1}: fix the board to match, then press the remote.")
                print(pos)
            if not presses.empty():
                presses.get()
                name = f"{len(rows) + 1:04d}.jpg"
                cv2.imwrite(str(out / name), grab_frame(cap, warmup=3), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                rows.append((name, pos.fen(), " ".join(steps[-1][1])))
                write_labels()
                print(f"saved {name}")
                if len(rows) == args.photos or not new_step():
                    print("\nGame finished." if len(rows) == args.photos else "\nGame over (no moves left).")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{len(rows)} labeled photos in {out}")


if __name__ == "__main__":
    main()
