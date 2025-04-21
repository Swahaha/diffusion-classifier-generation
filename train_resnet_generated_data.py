import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import datetime


# ResNet block definition-------------------------------------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

model = ResNet(num_classes=10)

# Dataset class----------------------------------------------------------------------------------------------------------------------
from torchvision.transforms import ToPILImage

class CombinedCIFAR10Dataset(Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path)
        self.images = data['images']  # Expect (N, 3, 32, 32) or (3072,)
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        # Handle flattened images
        if img.shape == (3072,):
            img = img.reshape(3, 32, 32)

        img = torch.tensor(img, dtype=torch.uint8)
        img = ToPILImage()(img)  # Convert to PIL for transforms

        if self.transform:
            img = self.transform(img)

        return img, label
    
# Transformers and dataloader setup-------------------------------------------------------------------------------------------------
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010)),
])

# Load dataset from .npz
COMBINED_DATA_PATH = "./combined_cifar10_with_diffusion_20250419_171722.npz"

# No transform yet — we apply them separately to train and val
full_dataset = CombinedCIFAR10Dataset(COMBINED_DATA_PATH, transform=None)

# 90/10 train/val split
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

# Assign different transforms
train_set.dataset.transform = transform_train
val_set.dataset.transform = transform_val

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 100

train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4)

# Training setup-------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# Training function-------------------------------------------------------------------------------------------------------------
def train_val(model, criterion, optimizer, train_loader, val_loader, device, EPOCHS=100, INITIAL_LR=0.1, STEP_SIZE=30, GAMMA=0.1):
    best_val_acc = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    CHECKPOINT_FOLDER = './resnet_checkpoint'
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    for i in range(EPOCHS):
        model.train()
        total_examples, correct_examples, train_loss = 0, 0, 0

        print(f"Epoch {i}:")
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_examples += targets.size(0)
            correct_examples += (predicted == targets).sum().item()
            train_loss += loss.item()

        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {correct_examples/total_examples:.4f}")

        # Validation
        model.eval()
        total_val, correct_val, val_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
                val_loss += loss.item()

        avg_val_acc = correct_val / total_val
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {avg_val_acc:.4f}")

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = os.path.join(CHECKPOINT_FOLDER, f'resnet_new_data_epoch_{i}.pth')
            torch.save({'state_dict': model.state_dict(), 'epoch': i}, save_path)
            print(f"Saved model to {save_path}")

        scheduler.step()
        print("")

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")

# Start training
train_val(model, criterion, optimizer, train_loader, val_loader, device, EPOCHS=150)
