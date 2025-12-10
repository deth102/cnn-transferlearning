import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

from PrepareData.Transform import transformation
from PrepareData.SignalSegments import (
    src_train, src_val, src_test,
    tgt_train, tgt_val, tgt_test
)

# ---------------- CONFIG ----------------
BATCH = 32
Fourier_transform = "STFT"   # hoặc "FFT"

# =====================================================
# 1) Lấy domain từ đường dẫn
# =====================================================
def get_domain_id(file_path):
    folder = os.path.basename(os.path.dirname(file_path)).lower()

    if folder == "0nm": return 0
    elif folder == "2nm": return 1
    elif folder == "4nm": return 2
    else:
        raise ValueError(f"Không nhận diện domain từ folder: {folder}")


# =====================================================
# 2) Dataset
# =====================================================
class VibDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe
        self.labels = sorted(self.df["label"].unique())
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- segment ----
        seg = row["segment"].astype(float)

        # ---- Fourier transform ----
        seg = transformation(seg, Fourier_transform)

        # ---- reshape ----
        if Fourier_transform == "FFT":
            seg = torch.tensor(seg, dtype=torch.float32).unsqueeze(0)  # (1,512)

        else:   # STFT
            seg = torch.abs(seg) if isinstance(seg, torch.Tensor) else torch.tensor(seg, dtype=torch.float32)
            seg = seg.unsqueeze(0)  # (1,F,T)

        # ---- label ----
        label = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)

        # ---- domain ----
        domain_id = torch.tensor(get_domain_id(row["path"]), dtype=torch.long)

        return seg, label, domain_id


# =====================================================
# 3) Build DataLoaders
# =====================================================
source_train_loader = DataLoader(VibDataset(src_train), batch_size=BATCH, shuffle=True)
source_val_loader   = DataLoader(VibDataset(src_val),   batch_size=BATCH, shuffle=False)
source_test_loader  = DataLoader(VibDataset(src_test),  batch_size=BATCH, shuffle=False)

target_train_loader = DataLoader(VibDataset(tgt_train), batch_size=BATCH, shuffle=True)
target_val_loader   = DataLoader(VibDataset(tgt_val),   batch_size=BATCH, shuffle=False)
target_test_loader  = DataLoader(VibDataset(tgt_test),  batch_size=BATCH, shuffle=False)