# multi_run_tinycnn.py

import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
NUM_RUNS        = 10
SAVE_THRESHOLD  = 0.75      # only save epochs where val_acc > 0.75
SAVE_TARGET     = 250       # number of saves per run
MAX_EPOCHS      = 500       # cap on epochs (to avoid infinite loop)
TRAIN_BATCH     = 128
VAL_BATCH       = 100
INITIAL_LR      = 0.005
STEP_SIZE       = 30
GAMMA           = 0.5
CHECKPOINT_ROOT = "./CNN_checkpoints"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

# Fixed validation transforms (no augmentation)
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])
val_set    = CIFAR10(root="./data", train=False, download=True, transform=transform_val)
val_loader = DataLoader(val_set, batch_size=VAL_BATCH, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ──────────────────────────────────────────────────────────────────────────────
# Training loop per run
# ──────────────────────────────────────────────────────────────────────────────
for run_id in range(NUM_RUNS):
    # seed everything for reproducibility
    seed = 60 + run_id
    random.seed(seed)
    torch.manual_seed(seed)

    # Build a “random flavor” of augmentations:
    # - horizontal flip with p ∈ [0.3,0.7]
    # - random crop padding ∈ {2,4,6}
    # - optional ColorJitter
    flip_p = random.uniform(0, 0.4)
    pad    = random.choice([2,4,6])
    # 50% chance to include color jitter
    if random.random() < 0.5:
        jitter = transforms.ColorJitter(
            brightness=random.uniform(0.1,0.3),
            contrast  =random.uniform(0.1,0.3),
            saturation=random.uniform(0.1,0.3),
            hue       =random.uniform(0.0,0.1)
        )
    else:
        jitter = None

    tr_augs = []
    tr_augs.append(transforms.RandomHorizontalFlip(p=flip_p))
    tr_augs.append(transforms.RandomCrop(32, padding=pad))
    if jitter:
        tr_augs.append(jitter)
    # ToTensor & Normalize last
    tr_augs.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2023,0.1994,0.2010))
    ])
    transform_train = transforms.Compose(tr_augs)

    # Datasets & loaders
    train_set    = CIFAR10(root="./data", train=True,  download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH, shuffle=True,  num_workers=4)

    # Prepare model, optimizer, scheduler
    model = TinyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=INITIAL_LR,
                          momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=STEP_SIZE,
                                          gamma=GAMMA)

    # Create run‐specific checkpoint folder
    run_folder = os.path.join(CHECKPOINT_ROOT, f"run_{run_id}")
    os.makedirs(run_folder, exist_ok=True)

    print(f"\n=== Starting run {run_id+1}/{NUM_RUNS} | flip_p={flip_p:.2f}, pad={pad}"
          + (f", jitter" if jitter else "") + " ===")

    saved_count = 0
    for epoch in range(1, MAX_EPOCHS+1):
        epoch_start = time.time()

        # --- train ---
        model.train()
        train_correct = 0
        train_total   = 0
        train_loss    = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds         = logits.argmax(dim=1)
            train_correct += (preds==labels).sum().item()
            train_total   += labels.size(0)

        train_loss /= train_total
        train_acc  = train_correct / train_total

        # --- validate ---
        model.eval()
        val_correct = 0
        val_total   = 0
        val_loss    = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)

                val_loss    += loss.item() * imgs.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds==labels).sum().item()
                val_total   += labels.size(0)

        val_loss /= val_total
        val_acc  = val_correct / val_total

        # step scheduler
        scheduler.step()

        # Logging
        epoch_time = time.time() - epoch_start
        print(f"Run {run_id+1}/{NUM_RUNS} | Epoch {epoch:4d} "
              f"| Train Loss {train_loss:.4f}, Acc {train_acc:.4f}"
              f" | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}"
              f" | Saved {saved_count}/{SAVE_TARGET}"
              f" | Time {epoch_time:.1f}s")

        # Save if above threshold
        if val_acc > SAVE_THRESHOLD and saved_count < SAVE_TARGET:
            ckpt_path = os.path.join(
                run_folder,
                f"4_run{run_id}_save{saved_count+1:03d}_epoch{epoch:04d}.pth"
            )
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_acc': val_acc
            }, ckpt_path)
            saved_count += 1

        # stop early if we've saved enough
        if saved_count >= SAVE_TARGET:
            print(f"→ Reached {SAVE_TARGET} saves for run {run_id}, moving on.\n")
            break

    if saved_count < SAVE_TARGET:
        print(f"⚠️  Only saved {saved_count} checkpoints for run {run_id} (less than {SAVE_TARGET})\n")
