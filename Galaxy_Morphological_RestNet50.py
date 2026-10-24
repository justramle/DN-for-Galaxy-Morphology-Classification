"""Train ResNet-50 (transfer learning) on the Galaxy10 DECals dataset."""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from astroNN.datasets import load_galaxy10
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from plotting import plot_accuracy_and_loss, plot_confusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 15

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "galaxy_r50_model.pth")


class Galaxy10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = Image.fromarray(img.astype(np.uint8))
            img = self.transform(img)
        return img, label


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base_model = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_data():
    images, labels = load_galaxy10()
    labels = to_categorical(labels, 10).astype(np.float32)
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.2)
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]


def build_dataloaders(train_images, train_labels, test_images, test_labels):
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = Galaxy10Dataset(train_images, train_labels, transform=transform_train)
    test_ds = Galaxy10Dataset(test_images, test_labels, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0

    with tqdm(loader, unit="batch", desc=f"Epoch {epoch + 1} Training") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_acc += acc.item()
            n_batches += 1
            pbar.set_postfix_str(
                f"train loss: {running_loss / n_batches:.4f} - train acc: {running_acc / n_batches:.4f}"
            )

    return running_loss / n_batches, running_acc / n_batches


def evaluate(model, loader, criterion, epoch):
    model.eval()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad(), tqdm(loader, unit="batch", desc=f"Epoch {epoch + 1} Validation") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

            running_loss += loss.item()
            running_acc += acc.item()
            n_batches += 1
            pbar.set_postfix_str(
                f"val loss: {running_loss / n_batches:.4f} - val acc: {running_acc / n_batches:.4f}"
            )

    return running_loss / n_batches, running_acc / n_batches, y_true, y_pred


def train(model, train_loader, test_loader, criterion, optimizer, scaler, patience=PATIENCE):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
    top1 = []
    counter, epoch = 0, 0
    val_y_true, val_y_pred = [], []

    while counter < patience:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
        val_loss, val_acc, val_y_true, val_y_pred = evaluate(model, test_loader, criterion, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        top1.append(val_acc)

        if val_acc >= max(top1[:-1], default=0):
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            counter = 0
        else:
            counter += 1

        epoch += 1

    print(f"\nTraining finished. Best top-1 accuracy: {max(top1):.4f}")
    return history, val_y_true, val_y_pred


def main():
    print(f"Using device: {DEVICE}")

    train_images, train_labels, test_images, test_labels = load_data()
    train_loader, test_loader = build_dataloaders(train_images, train_labels, test_images, test_labels)

    model = ResNet50().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()

    start = time.time()
    history, val_y_true, val_y_pred = train(model, train_loader, test_loader, criterion, optimizer, scaler)
    print(f"Total training time: {time.time() - start:.0f}s")

    val_y_true = [np.argmax(y) for y in val_y_true]
    print(f"Overall Accuracy: {accuracy_score(val_y_true, val_y_pred):.4f}")
    print(classification_report(val_y_true, val_y_pred))

    plot_accuracy_and_loss(
        history["train_acc"], history["val_acc"],
        history["train_loss"], history["val_loss"],
        "resnetcurve_accuracy.png", "resnetcurve_loss.png",
    )
    plot_confusion(val_y_true, val_y_pred, "resnet_confusion_matrix.png")


if __name__ == "__main__":
    main()
