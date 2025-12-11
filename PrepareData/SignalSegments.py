import os
import numpy as np
import pandas as pd
from PrepareData.PreProcessing import normalize_signal
# from PreProcessing import normalize_signal

# ---------------- CONFIG ----------------
BASE_DIR = "Vibration Data"
SEG_LEN  = 1024
OVERLAP  = 0
SEED = 42
np.random.seed(SEED)

SRC = "0nm"
TGT = "2nm"

# ---------------- HELPERS ----------------
def get_label(fname):
    f = fname.lower()
    for k in ["normal", "misalign", "unbalance", "bpfo", "bpfi"]:
        if k in f:
            return k.capitalize()
    return "Unknown"

def load_signal(path):
    return pd.read_csv(path, header=None).iloc[:, 0].values.astype(float)

def segment_signal(sig):
    step = SEG_LEN - OVERLAP
    return [sig[i:i+SEG_LEN] for i in range(0, len(sig)-SEG_LEN+1, step)]

def split_ratio(df, r1, r2, r3):
    df = df.sample(frac=1, random_state=SEED)
    n1 = int(len(df) * r1)
    n2 = int(len(df) * r2)
    return df[:n1], df[n1:n1+n2], df[n1+n2:]

# ---------------- LOAD DATA ----------------
all_records = []

for domain in sorted(os.listdir(BASE_DIR)):
    domain_path = os.path.join(BASE_DIR, domain)
    if not os.path.isdir(domain_path):
        continue

    for fname in os.listdir(domain_path):
        path = os.path.join(domain_path, fname)

        sig = normalize_signal(load_signal(path), "mean-std")
        label = get_label(fname)

        for idx, seg in enumerate(segment_signal(sig)):
            all_records.append([path, fname, idx, seg, label, domain])

df = pd.DataFrame(all_records, columns=["path", "file", "seg_idx", "segment", "label", "domain"])

# ---------------- SOURCE SPLIT (0nm) ----------------
df_src = df[df.domain.str.lower() == SRC].reset_index(drop=True)
src_train, src_val, src_test = split_ratio(df_src, 0.70, 0.15, 0.15)

# ---------------- TARGET SPLIT (2nm) ----------------
df_tgt = df[df.domain.str.lower() == TGT].reset_index(drop=True)
tgt_normal = df_tgt[df_tgt.label.str.lower() == "normal"]
tgt_fault  = df_tgt[df_tgt.label.str.lower() != "normal"]

tgt_norm_train, tgt_norm_val, tgt_norm_test = split_ratio(tgt_normal, 0.70, 0.15, 0.15)
tgt_fault_train, tgt_fault_val, tgt_fault_test = split_ratio(tgt_fault, 0.15, 0.15, 0.70)

tgt_train = pd.concat([tgt_norm_train, tgt_fault_train]).sample(frac=1, random_state=SEED)
tgt_val   = pd.concat([tgt_norm_val,  tgt_fault_val]).sample(frac=1, random_state=SEED)
tgt_test  = pd.concat([tgt_norm_test, tgt_fault_test]).sample(frac=1, random_state=SEED)

# ---------------- GLOBAL LABEL MAPPING ----------------
GLOBAL_LABELS = sorted(df.label.unique())
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(GLOBAL_LABELS)}

print("GLOBAL_LABELS:", GLOBAL_LABELS)
print("LABEL_TO_IDX:", LABEL_TO_IDX)

# ---------------- REPORT ----------------
print("SOURCE:", len(src_train), len(src_val), len(src_test))
print("\nTARGET SPLIT:")
print("Train:", len(tgt_train))
print("Val:",   len(tgt_val))
print("Test:",  len(tgt_test))



# =====================
# # 2) TARGET (2nm)
# # =====================

# # --- NORMAL → train + val + test ---
# tgt_normal = df_tgt[df_tgt.label.str.lower() == "normal"].reset_index(drop=True)

# tgt_train, tgt_val, tgt_test_norm = split(tgt_normal)

# # --- FAULT → test only ---
# tgt_fault = df_tgt[df_tgt.label.str.lower() != "normal"].reset_index(drop=True)

# # --- TEST = Normal-test + Fault-test ---
# tgt_test = pd.concat([tgt_test_norm, tgt_fault]).sample(frac=1, random_state=SEED).reset_index(drop=True)


# # =====================
# # SUMMARY
# # =====================
# print("SOURCE:", len(src_train), len(src_val), len(src_test))
# print("TARGET train (Normal):", len(tgt_train))
# print("TARGET val   (Normal):", len(tgt_val))
# print("TARGET test  (Normal + Fault):", len(tgt_test))


