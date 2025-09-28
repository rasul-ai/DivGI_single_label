import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import numpy as np

from models.get_model import get_arch
from utils.reproducibility import set_seeds

# ----------------------
# Config
# ----------------------
NUM_CLASSES = 23
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "./dataset"  # dataset with train/val/test
MODEL_PATH = "best_divgl_1.pth"

DO_MULTIGRANULARITIES = True
N_STEP = 4

# ----------------------
# Reproducibility
# ----------------------
set_seeds(42, torch.cuda.is_available())

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
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----------------------
# Data
# ----------------------
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_dataset = datasets.ImageFolder(root=f"{DATA_PATH}/test", transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ----------------------
# Inference
# ----------------------
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs, stage=1)

        if DO_MULTIGRANULARITIES:
            logits_list = outputs[0]
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            logits = outputs[0][-1]

        _, preds = torch.max(logits, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# ----------------------
# Metrics
# ----------------------
print("✅ Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4))

# Standard accuracy
acc = accuracy_score(all_labels, all_preds)

# Mean accuracy (average per-class recall)
class_acc = []
for c in range(NUM_CLASSES):
    idx = all_labels == c
    if np.sum(idx) > 0:
        class_acc.append(np.mean(all_preds[idx] == c))
macc = np.mean(class_acc)

# Balanced accuracy (sklearn handles it directly)
bacc = balanced_accuracy_score(all_labels, all_preds)

print(f"Overall Accuracy : {acc:.4f}")
print(f"Mean Accuracy (mAcc): {macc:.4f}")
print(f"Balanced Accuracy (bAcc): {bacc:.4f}")

# ----------------------
# Confusion Matrix
# ----------------------
cm = confusion_matrix(all_labels, all_preds, labels=np.arange(NUM_CLASSES))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)

plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Confusion Matrix - Test Set")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("📊 Confusion matrix saved as confusion_matrix.png")
