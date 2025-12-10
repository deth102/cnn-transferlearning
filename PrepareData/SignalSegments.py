import os
import numpy as np
import pandas as pd
from PrepareData.PreProcessing import normalize_signal


# ---------------- CONFIG ----------------
BASE_DIR = "Vibration Data"   # thư mục gốc chứa 0nm, 2nm, 4nm
SEG_LEN  = 1024
OVERLAP  = 0

TRAIN, VAL = 0.7, 0.15
SEED = 42
np.random.seed(SEED)

# -------------- HELPERS -----------------
def get_label(fname):
    f = fname.lower()
    for k in ["normal", "misalign", "unbalance", "bpfo", "bpfi"]:
        if k in f:
            return k.capitalize()
    return "Unknown"


def load_signal(path):
    return pd.read_csv(path, header=None).iloc[:, 0].values.astype(float)


def segment_signal(sig, seg_len, overlap):
    step = seg_len - overlap
    return [
        sig[i:i+seg_len]
        for i in range(0, len(sig) - seg_len + 1, step)
    ]


# -------------- LOAD ALL DOMAINS ----------------
all_records = []

# Ví dụ domain folder: 0nm, 2nm, 4nm
domain_folders = sorted(os.listdir(BASE_DIR))

for domain in domain_folders:
    domain_path = os.path.join(BASE_DIR, domain)

    if not os.path.isdir(domain_path):
        continue

    for fname in os.listdir(domain_path):
        full_path = os.path.join(domain_path, fname)

        label = get_label(fname)
        sig   = load_signal(full_path)

        # normalize toàn signal, không normalize từng segment
        sig = normalize_signal(sig, "mean-std")

        segs = segment_signal(sig, SEG_LEN, OVERLAP)

        for idx, seg in enumerate(segs):
            # Thêm path để DataLoader dùng nhận domain
            all_records.append([
                full_path,     # path
                fname,         # filename
                idx,           # segment index
                seg,           # segment data
                label,         # class label
                domain         # domain name (0nm / 2nm / 4nm)
            ])

df = pd.DataFrame(
    all_records,
    columns=["path", "file", "seg_idx", "segment", "label", "domain"]
)

# -------------- SPLIT DATA ----------------
# train, val, test = [], [], []

# for lb in df["label"].unique():
#     sub = df[df.label == lb].sample(frac=1, random_state=SEED)

#     n = len(sub)
#     n_tr  = int(n * TRAIN)
#     n_val = int(n * VAL)

#     train.append(sub[:n_tr])
#     val.append(sub[n_tr: n_tr + n_val])
#     test.append(sub[n_tr + n_val:])

# df đã chứa: path, file, seg_idx, segment, label, domain

# df chứa: path, file, seg_idx, segment, label, domain

# df chứa: path, file, seg_idx, segment, label, domain

SRC = "0nm"
TGT = "2nm"

df_src = df[df.domain.str.lower() == SRC].reset_index(drop=True)
df_tgt = df[df.domain.str.lower() == TGT].reset_index(drop=True)

# ===== Helper =====
def split(df_in):
    df_in = df_in.sample(frac=1, random_state=SEED)
    n = len(df_in)
    n_tr  = int(n * TRAIN)
    n_val = int(n * VAL)
    return df_in[:n_tr], df_in[n_tr:n_tr+n_val], df_in[n_tr+n_val:]


# =====================
# 1) SOURCE (0nm)
# =====================
src_train, src_val, src_test = split(df_src)


# =====================
# 2) TARGET (2nm)
# =====================

# --- NORMAL → train + val + test ---
tgt_normal = df_tgt[df_tgt.label.str.lower() == "normal"].reset_index(drop=True)

tgt_train, tgt_val, tgt_test_norm = split(tgt_normal)

# --- FAULT → test only ---
tgt_fault = df_tgt[df_tgt.label.str.lower() != "normal"].reset_index(drop=True)

# --- TEST = Normal-test + Fault-test ---
tgt_test = pd.concat([tgt_test_norm, tgt_fault]).sample(frac=1, random_state=SEED).reset_index(drop=True)


# =====================
# SUMMARY
# =====================
print("SOURCE:", len(src_train), len(src_val), len(src_test))
print("TARGET train (Normal):", len(tgt_train))
print("TARGET val   (Normal):", len(tgt_val))
print("TARGET test  (Normal + Fault):", len(tgt_test))



