# chess_vision

Chess board recognition from a photo of a physical board.
A fixed phone camera photographs the board, a fine-tuned ResNet18 classifies each of the 64 squares, and Stockfish gives the best move **for both sides**, since a photo can't show whose turn it is.

**Input:** one 4K frame of the board. **Output:** the position as a FEN string, plus Stockfish's best move and evaluation if White is to move and if Black is to move.

CS4337 Computer Vision, Texas State University, Fall 2026. Type 1 (Implementation) project, solo.

---

## 1. Pipeline

```
4K frame ──warp──> top-down board ──crop──> 64 squares ──ResNet18──> 64 x 13 scores
   (homography from 4 calibrated corners)  (lean-aware, 224x224)        │
                                                                         ▼
       best move for White + for Black <──Stockfish── FEN <──king rule── best class per square
```

| Step | What happens | Code |
|---|---|---|
| Calibration (once) | Click the 4 board corners and the two king tops: gives the homography plus the lean of tall pieces | `calibrate.py`, `board.py` |
| Warp | The perspective photo becomes a flat 1344×1344 top-down board (112 px per square) | `board.py` `warp()` |
| Crop | One crop per square, stretched toward where a tall piece's top leans at this camera angle, resized to 224×224 | `board.py` `crop_boxes()`, `crops()` |
| Classify | ResNet18 (ImageNet-pretrained, all layers fine-tuned) gives 13 class scores per square: empty plus 6 piece types × 2 colors | `train.py`, `recognize.py` |
| King rule | A legal board has exactly one king per color. A missing king goes to the square that looks most like one; extra kings fall back to their second-best class | `recognize.py` `fix_kings()` |
| FEN | The 64 labels become a FEN piece placement (python-chess) | `recognize.py` `to_fen()` |
| Engine | Stockfish at depth 18, run once with White to move and once with Black to move | `analyze.py` |

## 2. Setup

**Hardware** (the model only knows this setup):
- the green tournament board and pieces it was trained on
- iPhone 15 on a tripod at the **side of the board (a-file side)**, angled steeply down, close to overhead
- one overhead lamp
- the phone streams to the laptop with **Iriun Webcam**. OpenCV camera index 2 (`CAMERA_INDEX` in `camera.py`). Set Iriun's *Video format* to **3840×2160**, or frames come out at 1080p and the scripts stop with an error

**Software** (versions used):
- Python 3.12
- opencv-python 4.13, python-chess 1.11, numpy 2.4, Pillow 12.1, torch 2.11, torchvision 0.26
- Stockfish 19: `winget install Stockfish.Stockfish`. `analyze.py` finds it on PATH or in the winget folder; `--engine PATH` overrides that
- Google Colab (T4 GPU) for training, because the laptop has no NVIDIA GPU

```
pip install opencv-python python-chess numpy pillow torch torchvision
```

Run every command from the `chess_vision/` folder.

## 3. Steps

Each script's docstring explains it in more detail.

| # | Command | Input | Output |
|---|---|---|---|
| 1 | `python scripts/calibrate.py` | live frame of the board in the **starting position**; you click a1, h1, h8, a8, then the tops of the kings on e1 and e8 | `calibration.json` (corners, foot point, lean) |
| 2 | `python scripts/capture.py g01` (random game)<br>`python scripts/capture.py test01 --pgn pgn/test01.pgn` (real game) | live frames. The script shows the position to set up; a press of the Bluetooth remote saves the photo | `data/raw/<game>/0001.jpg ...` (raw 4K) + `labels.csv` (file, FEN) + `calibration.json` (the calibration this game was shot with; `build_dataset.py` and `recognize.py --eval` crop each game with its own, so recalibrating never breaks older data) |
| 3 | `python scripts/build_dataset.py`<br>then zip it: `python -c "import shutil; shutil.make_archive('outputs/crops', 'zip', 'data', 'crops')"` | `data/raw/` | `data/crops/{train,val,test}/<class>/<game>_<photo>_<square>.jpg` (ImageFolder layout) + class counts; `outputs/crops.zip` for Colab |
| 4 | `notebooks/train_colab.ipynb` on Colab, T4 GPU (upload `outputs/crops.zip` and `scripts/train.py` to `MyDrive/chess_vision/`, then Run all). Download `model.pt` from `MyDrive/chess_vision/outputs/` into `outputs/` on the laptop | `crops.zip`, `train.py` | `model.pt` (best validation epoch; `recognize.py` and `demo.py` load it from `outputs/model.pt`), `train_log.csv`, `training_curves.png`, `confusion_matrix.png`, `test_mistakes.png/.csv` |
| 5 | `python scripts/recognize.py photo.jpg`<br>`python scripts/recognize.py --eval data/raw/test01 data/raw/test02 data/raw/test03` | a 4K frame from the current camera position (uses the root `calibration.json`), or labeled game folders (each uses its own calibration) | the FEN and the board as text; with `--eval`, whole-board accuracy with and without the king rule |
| 6 | `python scripts/analyze.py "<FEN>"` | FEN piece placement | best move + evaluation for each side |
| 7 | `python scripts/demo.py`<br>`python scripts/demo.py --images data/raw/test02` (offline) | live camera, or a folder of saved frames (cropped with that folder's own `calibration.json`) | OpenCV window: live board on the left, analysis on the right |

**Demo keys:**
- `SPACE`: read the board and analyze it
- **left-click** a square: the model's next-best guess for it
- **right-click** a square: cycle through all 13 classes
- `s`: save the frame + corrected FEN to `data/raw/live_<date>/` as new training data
- `q`: quit

Saved `live_*` games always go to the training split, so rebuilding the dataset and retraining teaches the model the mistakes it made. Saving is disabled with `--images`, so test frames can't leak into training.

**Data split, by game:** photos one move apart are near-duplicates, so train, val and test never share a game.
- **train:** 6 random-move games, g03–g08 (plus any `live_*`)
- **val:** 2 random-move games, g01 and g02, drawn with seed 4337
- **test:** 2 real games replayed from PGN. test01 is Byrne–Fischer 1956 (41 photos); test02 is Kasparov–Topalov 1999 (44 photos)
- **test (added 2026-11-20):** test03, a 15-photo random game shot after the camera was moved and recalibrated, to check the model on the rebuilt setup. Every folder whose name starts with `test` goes to the test split

The model in section 4 was trained before test03 and `data/raw/live_2026-11-20/` (8 demo corrections) existed. Rebuilding the dataset now adds test03 to test (training unchanged) and the live corrections to train (training changes). To rebuild the exact v1 training set, move `live_2026-11-20/` out of `data/raw/` first.

**Training:**
- Adam, learning rate 1e-4, batch size 64, 10 epochs, cross-entropy loss
- augmentation: color jitter (brightness and contrast 0.3, saturation 0.2) and shifts up to 4%
- no flips, because lean direction and shadows are fixed by the camera

## 4. Results

On the test games, which the model never saw during training or epoch selection:

| Metric | Result |
|---|---|
| Per-square accuracy | **99.50%** (27 wrong of 5,440) |
| Baseline "always empty" | 64.32% |
| Whole-board accuracy, model only | 68.2% (58 of 85 photos) |
| Whole-board accuracy, with king rule | **100%** (85 of 85 photos) |
| Best validation accuracy | 99.90% (epoch 9) |

- **Every one of the 27 mistakes was a king.** 19 black kings were read as black bishops (test02) and 8 white kings as white queens (test01).
- Each of those squares is one a king never stood on in the training games: random play keeps kings near their home rank, and test02 has Topalov's king walk to a3–e1.
- Because of the steep camera angle, a piece's lean depends on its square, so a king seen from an unfamiliar angle looked like a bishop or queen to the model.
- Every wrong photo had exactly one wrong square, so the one-king-per-color rule repairs all of them.

The figures (`training_curves.png`, `confusion_matrix.png`, `test_mistakes.png`) are written to `outputs/` by the notebook.

## 5. Repo layout

```
chess_vision/
├── calibration.json        current camera calibration (corners, foot point, lean); each data/raw/<game>/ keeps its own copy
├── notebooks/
│   └── train_colab.ipynb   training + validation + test evaluation on Colab
├── pgn/                    the two real games used as the test set
└── scripts/
    ├── camera.py           opens Iriun at 4K, checks the resolution
    ├── remote.py           Bluetooth shutter remote as a capture trigger (Windows hotkey)
    ├── calibrate.py        click corners + king tops -> calibration.json
    ├── board.py            homography warp + lean-aware crops (used everywhere)
    ├── capture.py          labeled photo capture (random games or PGN replay)
    ├── build_dataset.py    raw frames -> labeled square crops, split by game
    ├── train.py            ResNet18 fine-tuning
    ├── recognize.py        frame -> FEN, king rule, --eval
    ├── analyze.py          FEN -> Stockfish best move for both sides
    └── demo.py             live OpenCV window with click-to-correct
```

**Dataset:** the 25,920 labeled crops the model was trained and tested on (g01–g08, test01, test02) are `crops.zip` on the [v1-dataset release](https://github.com/aidanorines/ChessVision/releases/tag/v1-dataset). Unzip it into `data/` to get `data/crops/{train,val,test}/<class>/`. Every game's labels (photo file, FEN, moves) are in git as `data/raw/<game>/labels.csv`. The trained v1 model is `model.pt` on the same release: put it in `outputs/`, where `recognize.py` and `demo.py` load it.

**Not in git** (see `.gitignore`):
- `data/raw/` photos: the raw 4K frames (625 MB), not published
- `*.pt`: the 45 MB trained model
- `outputs/`

The notebook recreates the model from `crops.zip`.

## 6. Limitations

- **One setup only.** Trained on one board, one set of pieces, one camera position and one lamp. Another board or piece set needs new training data. A moved camera needs `calibrate.py` again, and a very different angle or lighting may also need new data.
- **King coverage.** The model never saw kings on most of the board. The king rule repairs this, but only for kings. A misread of any other piece type wouldn't be caught. Fixes: corrections saved from the demo, or capture games where the kings travel.
- **Lighting.** Augmentation covers moderate brightness and contrast changes, but not a different color temperature, strong side shadows or glare.
- **No turn, castling or en passant from a photo.** Both sides are analyzed. Castling rights are inferred from whether kings and rooks stand on their start squares, and en passant is ignored.
- **Clear board needed.** Hands or pieces mid-move confuse the model. That's why the demo reads the board only when you press a key.
- **Small test set.** 85 photos from 2 games, so 100% whole-board accuracy shows the rule works on these games, not that the system never fails.
