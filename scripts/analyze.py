"""Ask Stockfish for the best move for BOTH sides of a recognized position.

Pipeline step: output (after recognize.py).  FEN piece placement --> best move + evaluation if White
is to move, and if Black is to move. A photo can't tell whose turn it is, so both are shown.

Decisions (2026-11-17, PROJECT_LOG.md):
  - fixed search depth (default 18): same answer every run, reproducible for the report
  - castling inferred: a right is kept only if that king and rook still stand on their start squares
  - impossible side skipped: if Black's king is in check it can't be White's turn (and vice versa)
  - output: best move + evaluation, from White's point of view (+ = White better, M3 = mate in 3)

Stockfish runs as a separate program; python-chess (chess.engine) talks to it over the UCI protocol.

Usage (from chess_vision/):
  python scripts/analyze.py "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"
"""
import argparse
import os
import shutil
from pathlib import Path

import chess
import chess.engine

DEPTH = 18
# winget installs Stockfish here without adding it to PATH
WINGET_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"


def find_stockfish():
    """Stockfish on PATH, else the winget install folder."""
    found = shutil.which("stockfish") or next(WINGET_DIR.glob("Stockfish.Stockfish*/**/stockfish*.exe"), None)
    if not found:
        raise SystemExit("Stockfish not found: install it (winget install Stockfish.Stockfish) or pass --engine PATH")
    return str(found)


def infer_castling(placement):
    """Castling rights the photo allows: king and rook on their start squares (can't see if they moved and came back)."""
    board = chess.Board(placement + " w - - 0 1")
    rights = ""
    for flag, king_sq, rook_sq, color in (("K", "e1", "h1", True), ("Q", "e1", "a1", True),
                                          ("k", "e8", "h8", False), ("q", "e8", "a8", False)):
        if (board.piece_at(chess.parse_square(king_sq)) == chess.Piece(chess.KING, color) and
                board.piece_at(chess.parse_square(rook_sq)) == chess.Piece(chess.ROOK, color)):
            rights += flag
    return rights or "-"


def format_score(score):
    """PovScore -> text from White's view: '+0.45', '-1.20', 'M3' (White mates in 3), '-M2' (Black mates in 2)."""
    s = score.white()
    if s.is_mate():
        return f"M{s.mate()}" if s.mate() > 0 else f"-M{-s.mate()}"
    return f"{s.score() / 100:+.2f}"   # centipawns -> pawns


def analyze(engine, placement, depth=DEPTH):
    """Best move for each side to move. Returns {"White": text, "Black": text}."""
    castling = infer_castling(placement)
    result = {}
    for color, letter in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        side, other = ("White", "Black") if color else ("Black", "White")
        board = chess.Board(f"{placement} {letter} {castling} - 0 1")
        board.clean_castling_rights()   # drop any right python-chess considers impossible
        status = board.status()
        if status & chess.STATUS_OPPOSITE_CHECK:
            result[side] = f"impossible: {other} is in check, so it can't be {side} to move"
        elif status != chess.STATUS_VALID:
            reasons = [f.name.lower().replace("_", " ") for f in chess.Status if f and f in status]   # e.g. "pawns on backrank"
            result[side] = "impossible: " + ", ".join(reasons)
        elif board.is_checkmate():
            result[side] = f"{side} is checkmated"
        elif board.is_stalemate():
            result[side] = "stalemate"
        else:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            result[side] = f"{board.san(info['pv'][0])}  ({format_score(info['score'])})"
    return result


def _self_test():
    """Castling inference and the impossible-side check, without the engine."""
    start = chess.STARTING_BOARD_FEN
    assert infer_castling(start) == "KQkq"
    assert infer_castling("4k3/8/8/8/8/8/8/4K2R") == "K"
    # Black king h8 mated by Qg7 (guarded by Kf6): White-to-move is impossible, Black-to-move is mate
    result = analyze(None, "7k/6Q1/5K2/8/8/8/8/8")   # neither side reaches the engine, so None is fine
    assert result["White"].startswith("impossible: Black is in check"), result
    assert result["Black"] == "Black is checkmated", result


def main():
    ap = argparse.ArgumentParser(description="Stockfish best move for both sides of a FEN piece placement.")
    ap.add_argument("placement", help='FEN piece placement, e.g. "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"')
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--engine", help="path to the Stockfish executable")
    args = ap.parse_args()

    _self_test()
    placement = args.placement.split()[0]   # accept a full FEN too; only the board part is used
    with chess.engine.SimpleEngine.popen_uci(args.engine or find_stockfish()) as engine:
        for side, text in analyze(engine, placement, args.depth).items():
            print(f"{side} to move: {text}")


if __name__ == "__main__":
    main()
