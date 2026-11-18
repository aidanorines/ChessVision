"""Train the square classifier: pretrained ResNet18, fine-tuned on our 13 classes.

Pipeline step: model training. Input: data/crops/{train,val}/<class>/*.jpg (from build_dataset.py).
Output (in --out, default outputs/):
  model.pt        best epoch by validation accuracy: {"state_dict", "classes"}
  train_log.csv   loss and accuracy per epoch (for the report's training curves)

Decisions (PROJECT_LOG.md): fine-tune ALL layers with a small learning rate; light augmentation
(color jitter + small shifts, no flips/rotations because lean and shadows are fixed by the camera);
10 epochs, Adam lr 1e-4; keep the epoch with the best validation accuracy.
The test split is NOT touched here; only the test cells in notebooks/train_colab.ipynb look at it, once, at the end.

Usage:
  Colab (GPU):  python train.py --data /content/data/crops --out /content/drive/MyDrive/chess_vision/outputs
  Local smoke test (CPU, a few images, checks the code runs):  python scripts/train.py --limit 64 --epochs 1 --workers 0
"""
import argparse
import csv
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics the pretrained weights expect

# Train-time augmentation: new random version of each crop every epoch, so near-duplicate crops
# (e.g. a king that never moved) don't look pixel-identical to the model.
TRAIN_TF = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),   # lamp / exposure drift, hand shadows
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),             # up to ~9 px shift: calibration error
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
# Validation sees crops exactly as they are, so its score is comparable between epochs.
EVAL_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def build_model(n_classes):
    """ResNet18 with ImageNet weights; the 1000-class final layer is replaced by one for our classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)   # 512 features -> 13 scores
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates. Returns (loss, accuracy)."""
    model.train(optimizer is not None)   # train mode: BatchNorm updates its statistics; eval mode: frozen
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            loss = loss_fn(scores, labels)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()    # gradients of the loss with respect to every weight
                optimizer.step()   # nudge every weight a little downhill
            total_loss += loss.item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
            seen += len(labels)
    return total_loss / seen, correct / seen


def main():
    ap = argparse.ArgumentParser(description="Fine-tune ResNet18 on the square crops.")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "crops")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, help="use only this many random images per split (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = datasets.ImageFolder(args.data / "train", TRAIN_TF)   # class = folder name, sorted alphabetically
    val_set = datasets.ImageFolder(args.data / "val", EVAL_TF)
    assert train_set.classes == val_set.classes, "train and val must have the same class folders"
    classes = train_set.classes
    if args.limit:
        rng = random.Random(0)
        train_set = Subset(train_set, rng.sample(range(len(train_set)), args.limit))
        val_set = Subset(val_set, rng.sample(range(len(val_set)), args.limit))

    train_loader = DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, args.batch, num_workers=args.workers)
    model = build_model(len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()   # plain loss for the baseline; the imbalance fix is decided after evaluating it
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   # all layers train (full fine-tune)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | {len(train_set)} train, {len(val_set)} val | {len(classes)} classes: {classes}")
    best = -1.0
    with open(args.out / "train_log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
            seconds = time.time() - start
            log.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{seconds:.0f}"])
            f.flush()
            saved = ""
            if val_acc > best:   # keep only the best epoch, so extra epochs can't make the saved model worse
                best = val_acc
                torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out / "model.pt")
                saved = "  <- saved"
            print(f"epoch {epoch:2}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}  ({seconds:.0f}s){saved}")
    print(f"best val acc {best:.4f}; model in {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
