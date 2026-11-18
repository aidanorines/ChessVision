"""Recognize the board position in one photo: frame -> FEN.

Pipeline step: inference (after training, before Stockfish).
  frame (4K) --warp--> board --crops--> 64 x 224x224 --ResNet18--> 64 x 13 logits --king rule--> FEN

King rule (decided 2026-11-17, PROJECT_LOG.md): a legal position has exactly one king of each color.
On the test games every mistake was a king read as a bishop/queen on a square kings never visited in
training. So after taking each square's best class:
  - no king of a color   -> the square with the highest score for that king becomes that king
  - two+ kings of a color -> keep the most king-like one, the others fall back to their second-best class
This uses chess knowledge to repair the model's output; it does not change the model.

Usage (from chess_vision/):
  python scripts/recognize.py data/raw/test02/0030.jpg          print the FEN for one or more photos
  python scripts/recognize.py --eval data/raw/test01 data/raw/test02
      whole-board accuracy with and without the king rule, against each folder's labels.csv
"""
import argparse
import csv
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from PIL import Image

import board as geo
from camera import ROOT
from train import build_model, EVAL_TF

MODEL_PATH = ROOT / "outputs" / "model.pt"
KINGS = ("white_king", "black_king")


def load_model(path=MODEL_PATH):
    """Rebuild ResNet18 and fill in the trained weights from model.pt. Returns (model, class names)."""
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(len(ckpt["classes"]), pretrained=False)   # weights come from the file, no ImageNet download
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["classes"]


def square_logits(frame, model, H, boxes):
    """Warp, crop and classify one frame. Returns (square names in FEN order a8..h1, 64 x 13 logits)."""
    tiles = geo.crops(geo.warp(frame, H), boxes)
    # OpenCV images are BGR; training crops were loaded by PIL as RGB, so convert or red/blue are swapped
    batch = torch.stack([EVAL_TF(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))) for _, t in tiles])
    with torch.no_grad():
        return [name for name, _ in tiles], model(batch).numpy()


def fix_kings(logits, classes):
    """Best class per square, then force exactly one king per color.
    Returns (labels, changes) where changes = [(square index, old label, new label), ...]."""
    labels = [classes[i] for i in logits.argmax(1)]
    changes = []
    for king in KINGS:   # extra kings first, so a square freed here can't be picked again below
        k = classes.index(king)
        kings = sorted((i for i in range(64) if labels[i] == king), key=lambda i: -logits[i, k])
        for i in kings[1:]:
            new = classes[int(np.argsort(-logits[i])[1])]   # its second-best class
            changes.append((i, labels[i], new))
            labels[i] = new
    for king in KINGS:
        if king not in labels:
            score = logits[:, classes.index(king)].copy()
            score[[i for i in range(64) if labels[i] in KINGS]] = -np.inf   # never overwrite the other king
            i = int(score.argmax())
            changes.append((i, labels[i], king))
            labels[i] = king
    return labels, changes


def to_fen(names, labels):
    """Square names + class labels -> FEN piece placement (the board part only; turn and castling are unknown)."""
    position = chess.Board(None)
    for name, label in zip(names, labels):
        if label != "empty":
            color, piece = label.split("_")   # e.g. "black", "king"
            position.set_piece_at(chess.parse_square(name),
                                  chess.Piece(chess.PIECE_NAMES.index(piece), color == "white"))
    return position.board_fen()


def evaluate(game_dirs, model, classes):
    """Whole-board accuracy with and without the king rule on labeled games (data/raw/<game>/labels.csv).
    Each game is warped and cropped with its own saved calibration, so old games still score right after recalibrating."""
    total = [0, 0, 0]   # photos, correct without rule, correct with rule
    for d in game_dirs:
        H, foot, lean = geo.load_calibration(geo.game_calibration_path(d))
        boxes = geo.crop_boxes(foot, lean)
        n = raw_ok = fixed_ok = 0
        for row in csv.DictReader(open(d / "labels.csv")):
            names, logits = square_logits(cv2.imread(str(d / row["file"])), model, H, boxes)
            true = chess.Board(row["fen"]).board_fen()
            raw = to_fen(names, [classes[i] for i in logits.argmax(1)])
            labels, changes = fix_kings(logits, classes)
            fixed = to_fen(names, labels)
            n, raw_ok, fixed_ok = n + 1, raw_ok + (raw == true), fixed_ok + (fixed == true)
            if changes or fixed != true:
                note = ", ".join(f"{names[i]} {old}->{new}" for i, old, new in changes) or "no change"
                print(f"  {d.name}/{row['file']}: {note}  [{'correct' if fixed == true else 'STILL WRONG'}]")
        print(f"{d.name}: {n} photos | without rule {raw_ok}/{n} ({raw_ok / n:.1%}) | with rule {fixed_ok}/{n} ({fixed_ok / n:.1%})")
        total = [total[0] + n, total[1] + raw_ok, total[2] + fixed_ok]
    n, raw_ok, fixed_ok = total
    print(f"ALL: {n} photos | without rule {raw_ok}/{n} ({raw_ok / n:.1%}) | with rule {fixed_ok}/{n} ({fixed_ok / n:.1%})")


def _self_test():
    """fix_kings on synthetic logits: a missing king gets added, an extra king gets removed."""
    classes = ["black_bishop", "black_king", "empty", "white_king", "white_queen"]
    logits = np.zeros((64, 5))
    logits[:, 2] = 5                     # every square empty...
    logits[0, 0], logits[0, 1] = 6, 5.5  # a8: bishop beats king (the real failure)
    logits[1, 1] = 4                     # b8: weaker black king score
    logits[60, 3] = 9                    # e1: white king
    logits[61, 3], logits[61, 4] = 8, 7  # f1: second white king, queen second-best
    labels, _ = fix_kings(logits, classes)
    assert labels[0] == "black_king" and labels[1] == "empty", labels[:2]
    assert labels[60] == "white_king" and labels[61] == "white_queen", labels[60:62]


def main():
    ap = argparse.ArgumentParser(description="Photo of the board -> FEN, with the one-king-per-color rule.")
    ap.add_argument("images", nargs="*", type=Path, help="4K frames to recognize")
    ap.add_argument("--eval", nargs="+", type=Path, metavar="GAME_DIR", help="labeled game folders to score")
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    args = ap.parse_args()
    if not args.images and not args.eval:
        ap.error("give image paths or --eval GAME_DIR ...")

    _self_test()
    model, classes = load_model(args.model)
    H, foot, lean = geo.load_calibration()   # current camera position, for new photos
    boxes = geo.crop_boxes(foot, lean)       # same 64 boxes for every frame: the camera is fixed

    if args.eval:
        evaluate(args.eval, model, classes)
    for path in args.images:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"can't read {path}")
        names, logits = square_logits(frame, model, H, boxes)
        labels, changes = fix_kings(logits, classes)
        for i, old, new in changes:
            print(f"king rule: {names[i]} {old} -> {new}")
        fen = to_fen(names, labels)
        print(f"{path.name}: {fen}")
        print(chess.Board(fen + " w - - 0 1"), "\n")   # letters, not unicode pieces: the Windows console can't print those


if __name__ == "__main__":
    main()
