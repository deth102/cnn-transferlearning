import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from PrepareData.DataLoader import (
    source_test_loader,
    target_test_loader
)

from Backbone.CNN1D import CNN1D
from Backbone.CNN2D import CNN2D

# ============================
# CONFIG
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Fourier_transform = "STFT"   # phải giống lúc train
MODEL_PATH = "best_model.pth"

# Lấy số lớp từ dataloader
NUM_CLASSES = len(source_test_loader.dataset.labels)
CLASS_NAMES = source_test_loader.dataset.labels


# ============================
# LOAD MODEL
# ============================
model = CNN2D(NUM_CLASSES) if Fourier_transform == "STFT" else CNN1D(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

criterion = nn.CrossEntropyLoss()


# ============================
# TEST FUNCTION
# ============================
def test_model(model, loader, domain_name="unknown"):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, d in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, _ = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples

    print(f"\n===== TEST RESULT: {domain_name} =====")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {avg_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # === Confusion Matrix ===
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix ({domain_name})")
    plt.tight_layout()
    plt.show()

    return avg_loss, avg_acc


# ============================
# RUN TESTS
# ============================
print("Running Source Test...")
source_loss, source_acc = test_model(model, source_test_loader, domain_name="SOURCE (0nm test)")

print("Running Target Test...")
target_loss, target_acc = test_model(model, target_test_loader, domain_name="TARGET (2nm normal+fault)")


print("\n============= SUMMARY =============")
print(f"Source Test Accuracy: {source_acc:.4f}")
print(f"Target Test Accuracy: {target_acc:.4f}")
print("===================================")
