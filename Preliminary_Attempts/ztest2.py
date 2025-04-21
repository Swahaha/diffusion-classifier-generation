# generate_and_save_best_tinycnn.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# 1) Redefine TinyCNN
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

# 2) Unflatten utility
def unflatten_state_dict(flat_vec, reference_state_dict):
    new_dict, idx = {}, 0
    for k, v in reference_state_dict.items():
        n = v.numel()
        new_dict[k] = flat_vec[idx:idx+n].view_as(v)
        idx += n
    return new_dict

# 3) Load your diffusion code
from weight_diffusion import WeightUNet, WeightDiffusion, load_checkpoint_weights

# 4) Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5) Load a well‑trained diffusion checkpoint
unet_ckpt = "diffusion_checkpoints/diffusion_epoch_45.pth"
# determine input_size
tiny_files = [f for f in os.listdir("./CNN_checkpoints/run_0/") if f.endswith(".pth")]
input_size = load_checkpoint_weights(os.path.join("./CNN_checkpoints/run_0/", tiny_files[0])).numel()

unet = WeightUNet(input_size).to(device)
unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
unet.eval()
diffusion = WeightDiffusion(unet)

# 6) Prepare CIFAR‑10 validation loader
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
val_loader = DataLoader(
    CIFAR10(root="./data", train=False, download=True, transform=transform_val),
    batch_size=100, shuffle=False, num_workers=4
)

criterion = nn.CrossEntropyLoss()

# 7) Sample & evaluate K candidates
K = 5
best_acc = 0.0
best_state_dict = None

for sample_i in range(K):
    # a) Reverse‑diffusion chain
    with torch.no_grad():
        x = torch.randn(1, input_size, device=device)
        for t in reversed(range(diffusion.timesteps)):
            x = diffusion.p_sample(x, torch.tensor([t], device=device))
        sampled = x.squeeze(0).cpu()

    # b) Build TinyCNN and load weights
    ref_model = TinyCNN()
    gen_sd = unflatten_state_dict(sampled, ref_model.state_dict())
    gen_model = TinyCNN().to(device)
    gen_model.load_state_dict(gen_sd)
    gen_model.eval()

    # c) Evaluate
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = gen_model(imgs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total
    print(f"Sample {sample_i+1}/{K}: Accuracy = {acc*100:.2f}%")

    # d) Keep best
    if acc > best_acc:
        best_acc = acc
        best_state_dict = gen_model.state_dict()

# 8) Save the best generator checkpoint
os.makedirs("generated_checkpoints", exist_ok=True)
out_path = f"generated_checkpoints/tinycnn_from_diffusion_acc_{best_acc*100:.2f}.pth"
torch.save({'state_dict': best_state_dict}, out_path)
print(f"\nSaved best generated TinyCNN (Acc {best_acc*100:.2f}%) to:\n  {out_path}")
