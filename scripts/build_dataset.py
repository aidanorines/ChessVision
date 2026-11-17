"""Turn labeled raw frames into a per-square image dataset.

Pipeline step: dataset building (between data collection and training). For every photo in data/raw/<game>/:
warp the 4K frame once, cut the 64 lean-aware crops (board.py), and label each crop from the FEN.

Output (torchvision ImageFolder layout, one folder per class):
  data/crops/train/<class>/<game>_<photo>_<square>.jpg
  data/crops/val/<class>/...
  data/crops/test/<class>/...      only once test games (data/raw/test*) exist
The file name keeps game, photo and square, so any misclassified crop can be traced back to its photo.

Split is BY GAME: photos one move apart are near-duplicates, so the same game never appears in two splits.
  test  = games named test* (real games replayed from PGN)
  val   = VAL_GAMES random games, drawn with a fixed seed (reproducible, chosen before seeing any results)
  train = the remaining random games

Usage (from chess_vision/):  python scripts/build_dataset.py
Delete data/crops first to rebuild (e.g. after shooting the test games or changing the crop method).
"""
import csv
import random
from collections import Counter

import chess
import cv2

import board as geo
from camera import ROOT

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "crops"
VAL_GAMES = 2
SEED = 4337           # fixed, so the val draw is the same every run
JPEG_QUALITY = 95


def class_name(piece):
    """Folder name for a square's content. Not FEN letters: Windows file names ignore case, so P and p would merge."""
    if piece is None:
        return "empty"
    return f"{'white' if piece.color else 'black'}_{chess.piece_name(piece.piece_type)}"   # e.g. white_pawn


def split_games():
    games = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    test = [g for g in games if g.startswith("test")]
    rest = [g for g in games if g not in test]
    val = sorted(random.Random(SEED).sample(rest, VAL_GAMES))
    return {"train": [g for g in rest if g not in val], "val": val, "test": test}


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists. Delete it to rebuild; nothing is overwritten.")
    H, foot, lean = geo.load_calibration()
    boxes = geo.crop_boxes(foot, lean)   # same 64 boxes for every photo: the camera is fixed
    splits = split_games()

    for split, games in splits.items():
        counts = Counter()
        for game in games:
            for row in csv.DictReader(open(RAW / game / "labels.csv")):
                frame = cv2.imread(str(RAW / game / row["file"]))
                position = chess.Board(row["fen"])
                for square, tile in geo.crops(geo.warp(frame, H), boxes):
                    # label by the crop's own square name, so crop order can never shift the labels
                    label = class_name(position.piece_at(chess.parse_square(square)))
                    folder = OUT / split / label
                    folder.mkdir(parents=True, exist_ok=True)
                    name = f"{game}_{row['file'][:-4]}_{square}.jpg"
                    cv2.imwrite(str(folder / name), tile, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    counts[label] += 1
            print(f"{split:5} {game} done")
        print(f"\n{split}: games {games}, {sum(counts.values())} crops")
        for label, n in counts.most_common():
            print(f"  {label:14}{n}")
        print()


if __name__ == "__main__":
    main()
