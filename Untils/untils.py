import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
)

# ============================================================
#  ACCURACY
# ============================================================
def accuracy(outputs, labels):
    """
    outputs: logits (B, num_classes)
    labels: ground truth (B)
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


# ============================================================
#  LOSS FUNCTION SELECTOR
# ============================================================
def get_loss(loss_name):
    loss_name = loss_name.lower()

    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "smoothl1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Loss '{loss_name}' is not supported!")


# ============================================================
#  OPTIMIZER SELECTOR
# ============================================================
def get_optimizer(opt_name, parameters, lr, weight_decay=0):
    opt_name = opt_name.lower()

    if opt_name == "sgd":
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "adam":
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{opt_name}' is not supported!")


# ============================================================
#  LR SCHEDULER SELECTOR
# ============================================================
def get_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "step":
        return StepLR(optimizer, step_size=kwargs.get("step_size", 10),
                      gamma=kwargs.get("gamma", 0.1))

    elif scheduler_name == "multistep":
        return MultiStepLR(optimizer, milestones=kwargs.get("milestones", [30, 60]),
                           gamma=kwargs.get("gamma", 0.1))

    elif scheduler_name == "exp":
        return ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.95))

    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get("t_max", 20))

    elif scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    else:
        raise ValueError(f"Scheduler '{scheduler_name}' is not supported!")


# ============================================================
#  EPOCH METRICS (train_loss, val_loss, train_acc, val_acc)
# ============================================================
def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss, total_acc, total_samples = 0, 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_acc  += accuracy(out, y) * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc  = total_acc  / total_samples
    return avg_loss, avg_acc


# ============================================================
#  SAVE MODEL
# ============================================================
# def save_model(model, path="checkpoint.pth"):
#     torch.save(model.state_dict(), path)
#     print(f"Model saved at {path}")
