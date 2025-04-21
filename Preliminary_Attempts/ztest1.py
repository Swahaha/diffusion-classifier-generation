import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# ─── 1) Define TinyCNN (must match your training code) ───
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ─── 2) Helper to load a checkpoint's weights as a flat vector (CPU only) ───
def load_checkpoint_weights_cpu(path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    parts = []
    for v in state_dict.values():
        parts.append(v.view(-1))
    return torch.cat(parts)

# ─── 3) Imports for your diffusion code ───
from red_weight_diffusion import WeightUNet, WeightDiffusion

# ─── 4) Set up device explicitly as cuda:0 or cpu ───
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── 5) Load one of your diffusion checkpoints ───
ckpt_path = "toy_diffusion_checkpoints/diffusion_epoch_129.pth"
# Determine input_size by loading any TinyCNN checkpoint
first_tiny = next(f for f in os.listdir("./CNN_checkpoints/run_0/") if f.endswith(".pth"))
input_size = load_checkpoint_weights_cpu(os.path.join("./CNN_checkpoints/run_0/", first_tiny)).numel()

# Instantiate U-Net and load its weights
unet = WeightUNet(input_size).to(device)
unet.load_state_dict(torch.load(ckpt_path, map_location=device))
unet.eval()

diffusion = WeightDiffusion(unet)

# ─── 6) Sample one set of weights ───
with torch.no_grad():
    x = torch.randn(1, input_size, device=device)
    for t in reversed(range(diffusion.timesteps)):
        x = diffusion.p_sample(x, torch.tensor([t], device=device))
    sampled = x.squeeze(0).cpu()

# ─── 7) Build a fresh TinyCNN and load sampled weights ───
ref_model = TinyCNN()
ref_sd = ref_model.state_dict()
gen_sd = {}
idx = 0
for k, v in ref_sd.items():
    numel = v.numel()
    gen_sd[k] = sampled[idx : idx + numel].view_as(v)
    idx += numel

# Load onto device
gen_model = TinyCNN().to(device)
# Compare only .type() to avoid cuda vs cuda:0 mismatch
for name, p in gen_model.state_dict().items():
    # dummy check that shapes match
    assert p.numel() == ref_sd[name].numel(), f"[ERROR] numel mismatch in {name}"
# Actually load
gen_model.load_state_dict(gen_sd)

# ─── 8) Prepare CIFAR-10 validation loader ───
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
val_set = CIFAR10(root="./data", train=False, download=True, transform=transform_val)
val_loader = DataLoader(val_set, batch_size=100, shuffle=False, num_workers=4)

# ─── 9) Evaluate ───
gen_model.eval()
y_true_list = []
y_pred_list = []
with torch.no_grad():
    for imgs, targets in tqdm(val_loader, desc="Validating sampled model"):
        imgs, targets = imgs.to(device), targets.to(device)
        logits = gen_model(imgs)
        preds = logits.argmax(dim=1)
        y_true_list.extend(targets.cpu().tolist())
        y_pred_list.extend(preds.cpu().tolist())

# Overall accuracy
correct = sum(yt == yp for yt, yp in zip(y_true_list, y_pred_list))
accuracy = correct / len(y_true_list)
print(f"\n=== Generated TinyCNN Validation ===")
print(f"Overall Accuracy: {accuracy*100:.2f}%")

# ─── 10) Detailed reporting ───
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# Label distributions
true_counts = Counter(y_true_list)
pred_counts = Counter(y_pred_list)
print("\n--- Label Distributions ---")
print("True labels:")
for i, name in enumerate(class_names):
    print(f"  {name:10s}: {true_counts[i]:5d}")
print("Predicted labels:")
for i, name in enumerate(class_names):
    print(f"  {name:10s}: {pred_counts[i]:5d}")

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_true_list, y_pred_list, target_names=class_names, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true_list, y_pred_list)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\n--- Confusion Matrix ---")
print(df_cm)

# Top-5 misclassifications
confusions = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j:
            confusions.append((cm[i,j], class_names[i], class_names[j]))
confusions.sort(reverse=True)
print("\n--- Top 5 Misclassifications ---")
for cnt, true_c, pred_c in confusions[:5]:
    print(f"  {true_c:10s} → {pred_c:10s}: {cnt} times")
