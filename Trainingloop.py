import torch
import torch.nn as nn

from PrepareData.DataLoader import (
    source_train_loader, source_val_loader, source_test_loader,
    target_train_loader, target_val_loader, target_test_loader
)

from Backbone.CNN1D import CNN1D
from Backbone.CNN2D import CNN2D
from Loss.MMD import domain_mmd_loss


# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Fourier_transform = "STFT"    # "FFT" hoặc "STFT"
NUM_CLASSES = len(source_train_loader.dataset.labels)

EPOCHS = 20
LR = 1e-3
LAMBDA_MMD = 0.5


# =====================================================
# MODEL
# =====================================================
model = CNN2D(NUM_CLASSES) if Fourier_transform == "STFT" else CNN1D(NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# =====================================================
# EVALUATION FUNCTION
# =====================================================
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0

    with torch.no_grad():
        for x, y, d in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, _ = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)

    return loss_sum / total, correct / total


# =====================================================
# TRAIN LOOP (SOURCE + TARGET)
# =====================================================
best_score = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    src_correct = tgt_correct = 0
    src_total   = tgt_total   = 0
    running_loss = 0

    # zip source + target để có batch song song
    for (xs, ys, ds), (xt, yt, dt) in zip(source_train_loader, target_train_loader):

        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        xt, yt = xt.to(DEVICE), yt.to(DEVICE)
        ds, dt = ds.to(DEVICE), dt.to(DEVICE)

        # ===== GỘP 2 DOMAIN =====
        x = torch.cat([xs, xt], dim=0)
        y = torch.cat([ys, yt], dim=0)
        domain = torch.cat([ds, dt], dim=0)

        optimizer.zero_grad()

        logits, features = model(x)

        # ===== TÁCH LẠI =====
        s_logits = logits[: len(xs)]
        t_logits = logits[len(xs):]

        s_feat = features[: len(xs)]
        t_feat = features[len(xs):]

        # ===== CLASSIFICATION LOSS =====
        cls_loss = criterion(s_logits, ys) + criterion(t_logits, yt)

        # ===== DOMAIN MMD LOSS =====
        mmd_loss = domain_mmd_loss(features, y, domain)

        # ===== TỔNG LOSS =====
        loss = cls_loss + LAMBDA_MMD * mmd_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # ===== METRICS: SOURCE TRAIN ACCURACY =====
        src_correct += (s_logits.argmax(1) == ys).sum().item()
        src_total   += ys.size(0)

        # ===== METRICS: TARGET TRAIN ACCURACY =====
        tgt_correct += (t_logits.argmax(1) == yt).sum().item()
        tgt_total   += yt.size(0)

    src_train_acc = src_correct / src_total
    tgt_train_acc = tgt_correct / tgt_total

    scheduler.step()

    # =====================================================
    # VALIDATION PHASE
    # =====================================================
    src_val_loss, src_val_acc = evaluate(model, source_val_loader)
    tgt_val_loss, tgt_val_acc = evaluate(model, target_val_loader)

    # =====================================================
    # LOG
    # =====================================================
    print(
        f"Epoch {epoch:02d} | "
        f"S_Train Acc: {src_train_acc:.4f} | "
        f"T_Train Acc: {tgt_train_acc:.4f} | "
        f"S_Val Acc: {src_val_acc:.4f} | "
        f"T_Val Acc: {tgt_val_acc:.4f} | "
        f"loss: {loss:.4f}|"
        f"MMD loss: {mmd_loss:.4f}"
    )

    # =====================================================
    # SAVE BEST BASED ON TARGET-VAL
    # =====================================================
    alpha = 0.5
    beta  = 0.5

    score = alpha * src_val_acc + beta * tgt_val_acc

    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✓ Saved best model (epoch {epoch})")


print("\nTraining finished.")
