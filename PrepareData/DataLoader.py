import os
import torch
from torch.utils.data import Dataset, DataLoader
from PrepareData.Transform import transformation
from PrepareData.SignalSegments import (
    src_train, src_val, src_test,
    tgt_train, tgt_val, tgt_test,
    LABEL_TO_IDX
)

# ========================
# CONFIG
# ========================
BATCH = 32
Fourier_transform = "STFT"   # hoặc "FFT"

DOMAIN_TO_ID = {"0nm": 0, "2nm": 1, "4nm": 2}


def get_domain_id(path):
    """Trích domain ID từ folder chứa tệp."""
    folder = os.path.basename(os.path.dirname(path)).lower()
    return DOMAIN_TO_ID.get(folder, -1)


# ========================
# Dataset class
# ========================
class VibDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        seg = transformation(row["segment"].astype(float), Fourier_transform)

        # ----- Convert to tensor -----
        if isinstance(seg, torch.Tensor):
            seg = seg.clone().detach().float()
        else:
            seg = torch.from_numpy(seg).float()
        if Fourier_transform == "STFT":
            seg = torch.abs(seg)
            if seg.dim() == 2 and seg.size(1) < 8:   # soft padding
                seg = torch.nn.functional.pad(seg, (0, 8 - seg.size(1)))
        seg = seg.unsqueeze(0)   # (1, F, T) hoặc (1, L)

        # ----- Label & Domain -----
        label  = torch.tensor(LABEL_TO_IDX[row["label"]], dtype=torch.long)
        domain = torch.tensor(get_domain_id(row["path"]), dtype=torch.long)

        return seg, label, domain


# ========================
# Build DataLoaders
# ========================
def build_loader(df, shuffle=False):
    return DataLoader(VibDataset(df), batch_size=BATCH, shuffle=shuffle)

source_train_loader = build_loader(src_train, shuffle=True)
source_val_loader   = build_loader(src_val)
source_test_loader  = build_loader(src_test)

target_train_loader = build_loader(tgt_train, shuffle=True)
target_val_loader   = build_loader(tgt_val)
target_test_loader  = build_loader(tgt_test)
