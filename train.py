import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

from models.get_model import get_arch
from utils.reproducibility import set_seeds

# ----------------------
# Config
# ----------------------
NUM_CLASSES = 23
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "/teamspace/studios/this_studio/dataset"

DO_MULTIGRANULARITIES = True
N_STEP = 4

# ----------------------
# Reproducibility
# ----------------------
set_seeds(42, torch.cuda.is_available())

# ----------------------
# Balanced MixUp
# ----------------------
def balanced_mixup(x, y, alpha=0.1):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_onehot = torch.nn.functional.one_hot(y, NUM_CLASSES).float()
    y_shuffled = torch.nn.functional.one_hot(y[index], NUM_CLASSES).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_shuffled

    return mixed_x, mixed_y

# ----------------------
# Model
# ----------------------
model, mean, std = get_arch(
    model_name="resnet50",
    n_classes=NUM_CLASSES,
    do_multigranularities=DO_MULTIGRANULARITIES,
    n_step=N_STEP,
    do_multilabel=False,
    number_multi_label=[NUM_CLASSES]
)
model = model.to(DEVICE)

# ----------------------
# Data
# ----------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = datasets.ImageFolder(root=f"{DATA_PATH}/train", transform=transform_train)
val_dataset   = datasets.ImageFolder(root=f"{DATA_PATH}/val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ----------------------
# Loss & Optimizer
# ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

# ----------------------
# CSV Logger Setup
# ----------------------
csv_file = "metrics_log.csv"
with open(csv_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss", "train_acc", "train_prec", "train_rec", "train_f1", "train_bacc",
        "val_loss", "val_acc", "val_prec", "val_rec", "val_f1", "val_bacc"
    ])

# ----------------------
# Training Loop
# ----------------------
best_acc = 0.0

for epoch in range(EPOCHS):
    # ----------------------
    # Train
    # ----------------------
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Apply Balanced MixUp
        inputs, labels_mix = balanced_mixup(inputs, labels, alpha=0.8)

        optimizer.zero_grad()
        outputs = model(inputs, stage=1)

        if DO_MULTIGRANULARITIES:
            logits_list = outputs[0]
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            logits = outputs[0][-1]

        # Loss with soft labels
        loss = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(logits, dim=1), labels_mix)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Predictions
        _, preds = torch.max(logits, 1)
        _, true_labels = torch.max(labels_mix, 1)  # approximate true labels

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(true_labels.cpu().numpy())

    train_loss = running_loss / len(train_dataset)
    train_acc  = accuracy_score(all_targets, all_preds)
    train_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    train_rec  = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    train_f1   = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    train_bacc = balanced_accuracy_score(all_targets, all_preds)

    # ----------------------
    # Validation
    # ----------------------
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs, stage=1)

            if DO_MULTIGRANULARITIES:
                logits_list = outputs[0]
                logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
            else:
                logits = outputs[0][-1]

            loss = criterion(logits, labels)
            val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(logits, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_loss /= len(val_dataset)
    val_acc  = accuracy_score(val_targets, val_preds)
    val_prec = precision_score(val_targets, val_preds, average="macro", zero_division=0)
    val_rec  = recall_score(val_targets, val_preds, average="macro", zero_division=0)
    val_f1   = f1_score(val_targets, val_preds, average="macro", zero_division=0)
    val_bacc = balanced_accuracy_score(val_targets, val_preds)

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, bAcc: {train_bacc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, bAcc: {val_bacc:.4f}")

    # ----------------------
    # Save metrics to CSV
    # ----------------------
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            train_loss, train_acc, train_prec, train_rec, train_f1, train_bacc,
            val_loss, val_acc, val_prec, val_rec, val_f1, val_bacc
        ])

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_divgl_1.pth")
        print("✅ Saved new best model")

print("Training finished. Best Val Acc = {:.2f}%".format(best_acc * 100))
