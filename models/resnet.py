# import necessary dependencies
import argparse
import os, sys
import time
import datetime
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
CHECKPOINT_FOLDER = '/content/drive/MyDrive/College/ece661/Project/saved_model'

# define the block class
class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Convolutions
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))

        # Adding shortcut connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet class
class ResNet(nn.Module):

  def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stages of residual blocks
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        # Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

  def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channels, out_channels, stride))
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
    
n_classes = 10
model = ResNet(num_classes=n_classes)
print(model)

# useful libraries
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),  # Normalize to CIFAR-10 stats
])

# Preprocessing function for validation/testing data (without augmentation)
transform_val = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))  # Normalize to CIFAR-10 stats
])

# do NOT change these
from tools.dataset import CIFAR10
from torch.utils.data import DataLoader

# Change this to adapt for our data
DATA_ROOT = "./data"
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 100

train_set = CIFAR10(
    root=DATA_ROOT,
    mode='train',
    download=True,
    transform=transform_train
)
val_set = CIFAR10(
    root=DATA_ROOT,
    mode='val',
    download=True,
    transform=transform_train
)

# construct dataloader
train_loader = DataLoader(
    train_set,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
val_loader = DataLoader(
    val_set,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# specify the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Deploying to: {device}")

import torch.nn as nn
import torch.optim as optim

# initial learning rate
INITIAL_LR = 0.1

# momentum for optimizer
MOMENTUM = 0.9

# L2 regularization strength
REG = 1e-4

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# Add optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)

def train_val(model, criterion, optimizer, train_loader, val_loader, device, EPOCHS=100, INITIAL_LR=0.01, STEP_SIZE=30, GAMMA=0.1):

  CHECKPOINT_FOLDER = '/content/drive/MyDrive/College/ece661/Project/saved_model'
  best_val_acc = 0
  current_learning_rate = INITIAL_LR

  # Define learning rate scheduler
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

  print("==> Training starts!")
  print("="*50)
  for i in range(0, EPOCHS):
      # switch to train mode
      model.train()

      print("Epoch %d:" %i)
      # this help you compute the training accuracy
      total_examples = 0
      correct_examples = 0

      train_loss = 0 # track training loss if you want

      # Train the model for 1 epoch.
      for batch_idx, (inputs, targets) in enumerate(train_loader):
          # copy inputs to device
          inputs = inputs.to(device)
          targets = targets.to(device)

          # compute the output and loss
          outputs = model(inputs)
          loss = criterion(outputs, targets)

          # zero the gradient
          optimizer.zero_grad()

          # backpropagation
          loss.backward()

          # apply gradient and update the weights
          optimizer.step()

          # count the number of correctly predicted samples in the current batch
          _, predicted = torch.max(outputs.data, 1)
          total_examples += targets.size(0)
          correct_examples += (predicted == targets).sum().item()

          train_loss += loss.item()

      avg_loss = train_loss / len(train_loader)
      avg_acc = correct_examples / total_examples
      print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))

      # Validate on the validation dataset
      # switch to eval mode
      model.eval()

      # this help you compute the validation accuracy
      total_examples = 0
      correct_examples = 0

      # Track validation loss
      val_loss = 0

      # disable gradient during validation, which can save GPU memory
      with torch.no_grad():
          for batch_idx, (inputs, targets) in enumerate(val_loader):
              # copy inputs to device
              inputs = inputs.to(device)
              targets = targets.to(device)

              # compute the output and loss
              outputs = model(inputs)
              loss = criterion(outputs, targets)

              # count the number of correctly predicted samples in the current batch
              _, predicted = torch.max(outputs.data, 1)
              total_examples += targets.size(0)
              correct_examples += (predicted == targets).sum().item()

              val_loss += loss.item()

      # Calculate average loss and average accuracy
      avg_loss = val_loss / len(val_loader)
      avg_acc = correct_examples / total_examples
      print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))

      # save the model checkpoint
      if avg_acc > best_val_acc:
          best_val_acc = avg_acc
          if not os.path.exists(CHECKPOINT_FOLDER):
            os.makedirs(CHECKPOINT_FOLDER)
          print("Saving ...")
          state = {'state_dict': model.state_dict(),
                  'epoch': i,
                  'lr': current_learning_rate}
          save_path = os.path.join(CHECKPOINT_FOLDER, 'resnet_epoch_{}.pth'.format(i))

          torch.save(state, save_path)

      # Step learning rate scheduler
      scheduler.step()
      print(f"Updated Learning Rate: {scheduler.get_last_lr()[0]}\n")

      print('')

  print("="*50)
  print(f"==> Optimization finished! Best validation accuracy: {best_val_acc:.4f}")
  
  train_val(model, criterion, optimizer, train_loader, val_loader, device, EPOCHS=150, INITIAL_LR=INITIAL_LR)