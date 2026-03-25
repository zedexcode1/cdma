#!/usr/bin/env python3
"""
CDMA Replication -- Experiment 2 (pooled k-fold evaluation)
============================================================
Standalone script for running on a remote cloud Linux machine.

CHANGE FROM EXPERIMENT 1:
  Instead of computing metrics per fold and averaging (which inflates
  variance due to fold imbalance), this version pools all predictions
  across k folds for each repetition and computes ONE set of metrics
  over all ~110 participants. The standard deviation across reps then
  reflects only random initialization variance, not fold composition.

Usage:
    # Run all 7 conditions:
    python cdma_experiment.py --conditions all

    # Run specific conditions:
    python cdma_experiment.py --conditions ba1_rt ba1_it full_cdma

    # Run with custom data directory:
    python cdma_experiment.py --data-dir /path/to/data --conditions all

    # Push results to GitHub after completion:
    python cdma_experiment.py --conditions all --push

Output files:
    results/fold_predictions.csv  -- per-fold raw predictions (for resume)
    results/pooled_results.csv    -- one row per (condition, rep), pooled metrics
"""

import os
import sys
import time
import zipfile
import logging
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gdown


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters (Tao 2023, Section 7.4) -- DO NOT CHANGE
# ---------------------------------------------------------------------------
FEATURE_DIM   = 32
LSTM_HIDDEN   = 32
FRAME_SIZE    = 128
FRAME_STEP    = 64
EPOCHS        = 300      # no early stopping
LEARNING_RATE = 1e-3
BATCH_SIZE    = 16       # thesis value
N_REPS        = 10
THRESHOLD     = 0.5

# Will be set after argument parsing
DEVICE        = None
FEATURES_RT   = None
FEATURES_IT   = None
FOLD_FILE     = None
RESULTS_DIR   = None


# ---------------------------------------------------------------------------
# Google Drive file ID for the data zip
# ---------------------------------------------------------------------------
GDRIVE_FILE_ID = "1LJenZ-VXktBbroTI3btVSkRRq-glSWCb"


# ===========================================================================
# Data download and extraction
# ===========================================================================
def download_and_extract_data(data_dir):
    """
    Download features zip from Google Drive and extract.
    Skips download if data already exists (resume-safe).

    The zip must contain:
      cdma_features/rt/*_frames.npy   (110 files)
      cdma_features/it/*_frames.npy   (110 files)
      fold-lists.csv
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if already extracted
    if (os.path.exists(os.path.join(data_dir, "cdma_features"))
            and os.path.exists(os.path.join(data_dir, "fold-lists.csv"))):
        logger.info(f"Data already extracted at {data_dir}")
        return data_dir

    zip_path = os.path.join(data_dir, "data.zip")
    logger.info("Downloading data.zip from Google Drive...")
    gdown.download(id=GDRIVE_FILE_ID, output=zip_path, quiet=False)

    logger.info("Extracting data.zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    # Handle case where zip extracted into a subdirectory
    if not os.path.exists(os.path.join(data_dir, "fold-lists.csv")):
        for subdir in os.listdir(data_dir):
            candidate = os.path.join(data_dir, subdir)
            if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "fold-lists.csv")
            ):
                logger.info(f"Data found in subdirectory: {candidate}")
                return candidate

    return data_dir


# ===========================================================================
# Fold parsing
# ===========================================================================
def parse_fold_lists():
    """
    Parse fold-lists.csv.
    Structure:
      - header=None (first row has section labels 'Read'/'Interview')
      - row 1: fold labels (fold1...fold5)
      - rows 2+: participant IDs (may have single quotes)
      - RT folds: columns 0-4, IT folds: columns 7-11
    We use RT fold assignments for CV.
    """
    df = pd.read_csv(FOLD_FILE, header=None)
    fold_label_row = next(
        i for i, row in df.iterrows()
        if any(
            str(v).strip().strip("'\"") == "fold1"
            for v in row
            if pd.notna(v)
        )
    )

    def extract_folds(start_col, n_folds=5):
        result = {}
        for i in range(n_folds):
            col = start_col + i
            if col >= len(df.columns):
                break
            pids = [
                str(df.iloc[r, col]).strip().strip("'\"").strip()
                for r in range(fold_label_row + 1, len(df))
                if not pd.isna(df.iloc[r, col])
            ]
            result[i + 1] = [p for p in pids if p and p != "nan"]
        return result

    return {"read": extract_folds(0), "interview": extract_folds(7)}


def build_participant_info():
    """
    Build participant list and labels from frame files that actually exist.
    Labels from participant ID code: CF/CM=0 (control), PF/PM=1 (patient).
    """
    rt_pids = {
        f.stem.replace("_frames", "")
        for f in Path(FEATURES_RT).glob("*_frames.npy")
    }
    it_pids = {
        f.stem.replace("_frames", "")
        for f in Path(FEATURES_IT).glob("*_frames.npy")
    }
    both = rt_pids & it_pids
    label_map = {}
    for pid in both:
        parts = pid.split("_")
        if len(parts) >= 2:
            code = parts[1][:2].upper()
            if code in ("CF", "CM"):
                label_map[pid] = 0
            elif code in ("PF", "PM"):
                label_map[pid] = 1
    return sorted(label_map.keys()), label_map


# ===========================================================================
# Dataset and DataLoader
# ===========================================================================
class FeatureNormalizer:
    """
    Z-score normalizer fit on training participants only.
    Applied to test set using training statistics -- no leakage.
    """
    def fit(self, train_ids):
        vecs = [
            np.load(str(Path(folder) / f"{pid}_frames.npy")).reshape(-1, FEATURE_DIM)
            for pid in train_ids
            for folder in [FEATURES_RT, FEATURES_IT]
            if (Path(folder) / f"{pid}_frames.npy").exists()
        ]
        v = np.vstack(vecs)
        self.mean = v.mean(axis=0)
        self.std = v.std(axis=0)
        return self

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-8)


class AndroidsDataset(Dataset):
    """
    Loads all participant data into memory at init.
    ~600MB for full training set.
    """
    def __init__(self, pids, label_map, normalizer=None):
        self.pids = pids
        self.data = {}
        for pid in pids:
            rt = np.load(f"{FEATURES_RT}/{pid}_frames.npy").astype(np.float32)
            it = np.load(f"{FEATURES_IT}/{pid}_frames.npy").astype(np.float32)
            if normalizer:
                rt = normalizer.transform(rt)
                it = normalizer.transform(it)
            self.data[pid] = {
                "rt_frames": torch.from_numpy(rt),
                "it_frames": torch.from_numpy(it),
                "label": torch.tensor(label_map[pid], dtype=torch.long),
                "pid": pid,
            }

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return self.data[self.pids[idx]]


def collate_fn(batch):
    B = len(batch)
    n_rt = [b["rt_frames"].shape[0] for b in batch]
    n_it = [b["it_frames"].shape[0] for b in batch]
    N, L = max(n_rt), max(n_it)
    rt_pad = torch.zeros(B, N, FRAME_SIZE, FEATURE_DIM)
    it_pad = torch.zeros(B, L, FRAME_SIZE, FEATURE_DIM)
    rt_mask = torch.zeros(B, N)
    it_mask = torch.zeros(B, L)
    labels = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        rt_pad[i, :n_rt[i]] = b["rt_frames"]
        it_pad[i, :n_it[i]] = b["it_frames"]
        rt_mask[i, :n_rt[i]] = 1.0
        it_mask[i, :n_it[i]] = 1.0
        labels[i] = b["label"]
    return {
        "rt_frames": rt_pad,
        "it_frames": it_pad,
        "rt_mask": rt_mask,
        "it_mask": it_mask,
        "n_rt": torch.tensor(n_rt, dtype=torch.long),
        "n_it": torch.tensor(n_it, dtype=torch.long),
        "labels": labels,
    }


def get_dataloaders(fold, all_pids, label_map, fold_to_pids):
    test_ids = [
        p for p in fold_to_pids[fold]
        if Path(f"{FEATURES_RT}/{p}_frames.npy").exists()
        and Path(f"{FEATURES_IT}/{p}_frames.npy").exists()
    ]
    train_ids = [
        p for f, ids in fold_to_pids.items() if f != fold
        for p in ids
        if Path(f"{FEATURES_RT}/{p}_frames.npy").exists()
        and Path(f"{FEATURES_IT}/{p}_frames.npy").exists()
    ]
    assert not (set(test_ids) & set(train_ids)), "Person independence violated"
    norm = FeatureNormalizer().fit(train_ids)
    train_loader = DataLoader(
        AndroidsDataset(train_ids, label_map, norm),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        AndroidsDataset(test_ids, label_map, norm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, test_loader


# ===========================================================================
# Model
# ===========================================================================
class ITMLALayer(nn.Module):
    """IT-MLA -- Eq. 6.1, 6.2. Zero parameters."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True).expand_as(x)
        cos = F.cosine_similarity(x, a, dim=2)  # (B, M)
        return x * (1.0 + cos.unsqueeze(2))       # (B, M, D)


class LSTM1Layer(nn.Module):
    """Frame-level LSTM. Returns context (B, H) and probability (B, 1)."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(FEATURE_DIM, LSTM_HIDDEN, batch_first=True)
        self.head = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        ctx = out.mean(dim=1)
        prob = torch.sigmoid(self.head(ctx))
        return ctx, prob


class LSTM2Layer(nn.Module):
    """Sequence-level LSTM with masking. Returns context (B, H) and probability (B, 1)."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(LSTM_HIDDEN, LSTM_HIDDEN, batch_first=True)
        self.head = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x, lengths, mask):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True)
        N = out.size(1)
        m = mask[:, :N].unsqueeze(2).float()
        ctx = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        prob = torch.sigmoid(self.head(ctx))
        return ctx, prob


class CTGALayer(nn.Module):
    """CT-GA -- Eq. 7.2, 7.3. Zero parameters."""
    def __init__(self):
        super().__init__()

    def forward(self, C, O, mask_c, mask_o):
        def masked_mean(X, mask):
            m = mask.unsqueeze(2).float()
            return (X * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        r = masked_mean(C, mask_c)
        s = masked_mean(O, mask_o)

        def attend(X, ref, mask):
            cos = F.cosine_similarity(X, ref.unsqueeze(1).expand_as(X), dim=2)
            cos = cos.masked_fill(mask == 0, -1e9)
            w = F.softmax(cos, dim=1)
            return X * (1.0 + w.unsqueeze(2))

        C_star = attend(C, s, mask_c)
        O_star = attend(O, r, mask_o)
        r_star = masked_mean(C_star, mask_c)
        s_star = masked_mean(O_star, mask_o)
        return C_star, O_star, r, s, r_star, s_star


class CTFLayer(nn.Module):
    """CTF -- Eq. 7.4, 7.5."""
    def __init__(self):
        super().__init__()
        self.head_f = nn.Linear(LSTM_HIDDEN, 1)
        self.head_f_star = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, r, s, r_star, s_star):
        return (
            torch.sigmoid(self.head_f(r + s)),
            torch.sigmoid(self.head_f_star(r_star + s_star)),
        )


class CDMAModel(nn.Module):
    VALID_MODES = (
        "ba1_rt", "ba1_it",
        "itmla_rt", "itmla_it",
        "ctga_rt", "ctga_it",
        "full_cdma",
    )

    def __init__(self, mode="full_cdma"):
        super().__init__()
        assert mode in self.VALID_MODES
        self.mode = mode
        self.it_mla = ITMLALayer()
        self.lstm1_rt = LSTM1Layer()
        self.lstm1_it = LSTM1Layer()
        self.ct_ga = CTGALayer()
        self.lstm2_rt = LSTM2Layer()
        self.lstm2_it = LSTM2Layer()
        self.ctf = CTFLayer()

    def forward(self, batch):
        rt = batch["rt_frames"]
        it = batch["it_frames"]
        mc = batch["rt_mask"]
        mo = batch["it_mask"]
        n_rt = batch["n_rt"]
        n_it = batch["n_it"]

        B, N, M, D = rt.shape
        L = it.shape[1]
        H = LSTM_HIDDEN

        use_mla = self.mode not in ("ba1_rt", "ba1_it")
        use_ctga = self.mode in ("ctga_rt", "ctga_it", "full_cdma")
        use_ctf = self.mode == "full_cdma"
        # CT-GA needs both streams even if only one is reported in output
        need_rt = self.mode in ("ba1_rt", "itmla_rt", "ctga_rt", "full_cdma") or use_ctga
        need_it = self.mode in ("ba1_it", "itmla_it", "ctga_it", "full_cdma") or use_ctga

        probs = {}

        if need_rt:
            rf = rt.view(B * N, M, D)
            if use_mla:
                rf = self.it_mla(rf)
            c_flat, p_flat = self.lstm1_rt(rf)
            C = c_flat.view(B, N, H)
            p = p_flat.view(B, N)
            if self.mode in ("ba1_rt", "itmla_rt", "ctga_rt", "full_cdma"):
                nv = mc.sum(1).clamp(min=1.0)
                probs["p_c"] = (p * mc).sum(1, keepdim=True) / nv.unsqueeze(1)

        if need_it:
            of_ = it.view(B * L, M, D)
            if use_mla:
                of_ = self.it_mla(of_)
            o_flat, q_flat = self.lstm1_it(of_)
            O = o_flat.view(B, L, H)
            q = q_flat.view(B, L)
            if self.mode in ("ba1_it", "itmla_it", "ctga_it", "full_cdma"):
                nv = mo.sum(1).clamp(min=1.0)
                probs["p_o"] = (q * mo).sum(1, keepdim=True) / nv.unsqueeze(1)

        if use_ctga:
            C_star, O_star, r, s, r_star, s_star = self.ct_ga(C, O, mc, mo)
            _, probs["p_t"] = self.lstm2_rt(C_star, n_rt, mc)
            _, probs["p_d"] = self.lstm2_it(O_star, n_it, mo)

        if use_ctf:
            probs["p_f1"], probs["p_f2"] = self.ctf(r, s, r_star, s_star)

        # Eq. 7.6: mean of all available probabilities
        probs["p_hat"] = torch.stack(list(probs.values())).mean(dim=0)
        return probs


# ===========================================================================
# Loss
# ===========================================================================
class CombinedBCELoss(nn.Module):
    """Eq. 7.7: BCE averaged over all probability outputs except p_hat."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, outputs, labels):
        y = labels.float().unsqueeze(1)
        return torch.stack(
            [self.bce(v, y) for k, v in outputs.items() if k != "p_hat"]
        ).mean()


# ===========================================================================
# Training and evaluation
# ===========================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        bd = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        loss = criterion(model(bd), bd["labels"])
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def evaluate_model(model, loader, device):
    """
    Returns raw predictions and labels (not metrics).
    Metrics are computed later after pooling across folds.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            bd = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            p = model(bd)["p_hat"].squeeze(1).cpu()
            preds.extend((p > THRESHOLD).long().tolist())
            targets.extend(batch["labels"].cpu().tolist())
    return preds, targets


def compute_metrics(preds, targets):
    """Compute all metrics from raw predictions and labels."""
    return {
        "accuracy": 100.0 * accuracy_score(targets, preds),
        "precision": 100.0 * precision_score(targets, preds, zero_division=0),
        "recall": 100.0 * recall_score(targets, preds, zero_division=0),
        "f1": 100.0 * f1_score(targets, preds, zero_division=0),
    }


def train_and_evaluate(mode, fold, rep, device, all_pids, label_map, fold_to_pids):
    """Full train+eval for one (mode, fold, rep). Returns (preds, targets, elapsed_s)."""
    torch.manual_seed(rep * 42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rep * 42)

    train_loader, test_loader = get_dataloaders(fold, all_pids, label_map, fold_to_pids)
    model = CDMAModel(mode).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    criterion = CombinedBCELoss()

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            logger.info(
                f"  [{mode} F{fold} R{rep}] Epoch {epoch:3d}/{EPOCHS} loss={loss:.4f}"
            )

    elapsed = time.time() - t0
    preds, targets = evaluate_model(model, test_loader, device)
    return preds, targets, elapsed


# ===========================================================================
# Run one condition
# ===========================================================================
def run_condition(condition, all_pids, label_map, fold_to_pids, k_folds):
    """
    Run one condition across all folds and reps.

    KEY CHANGE (Experiment 2 / Prof. Vinciarelli):
    Instead of computing metrics per fold and averaging, we pool all raw
    predictions across the k folds for each rep, then compute ONE set of
    metrics over all ~110 participants per rep. This eliminates the variance
    caused by fold imbalance.

    We save two files:
      1) fold_predictions.csv  -- per-fold raw predictions (for resume)
      2) pooled_results.csv    -- one row per (condition, rep) with pooled metrics
    """
    pred_file = os.path.join(RESULTS_DIR, "fold_predictions.csv")
    pooled_file = os.path.join(RESULTS_DIR, "pooled_results.csv")

    # Load existing per-fold predictions for resume
    if os.path.exists(pred_file):
        pred_df = pd.read_csv(pred_file)
        done_folds = set(
            zip(pred_df["condition"], pred_df["fold"], pred_df["rep"])
        )
    else:
        pred_df = pd.DataFrame()
        done_folds = set()

    # Load existing pooled results for resume
    if os.path.exists(pooled_file):
        pooled_df = pd.read_csv(pooled_file)
        done_reps = set(
            zip(pooled_df["condition"], pooled_df["rep"])
        )
    else:
        pooled_df = pd.DataFrame()
        done_reps = set()

    already = sum(1 for r in done_reps if r[0] == condition)
    logger.info(f"=== {condition} === {already}/{N_REPS} reps fully pooled ===")

    t_cond = time.time()
    for rep in range(N_REPS):
        if (condition, rep) in done_reps:
            continue

        # Collect predictions from all folds for this rep
        all_preds_this_rep = []
        all_targets_this_rep = []
        total_elapsed = 0.0

        for fold in range(1, k_folds + 1):
            # Check if this specific fold was already run (resume support)
            if (condition, fold, rep) in done_folds:
                # Retrieve saved predictions for this fold
                mask = (
                    (pred_df["condition"] == condition)
                    & (pred_df["fold"] == fold)
                    & (pred_df["rep"] == rep)
                )
                saved = pred_df[mask]
                all_preds_this_rep.extend(saved["prediction"].tolist())
                all_targets_this_rep.extend(saved["target"].tolist())
                logger.info(f"  Fold {fold} rep {rep} loaded from cache ({len(saved)} pids)")
                continue

            logger.info(f"Running fold={fold} rep={rep}...")
            try:
                preds, targets, elapsed = train_and_evaluate(
                    condition, fold, rep, DEVICE,
                    all_pids, label_map, fold_to_pids,
                )
                total_elapsed += elapsed

                # Save per-fold predictions for resume
                fold_rows = []
                test_ids = [
                    p for p in fold_to_pids[fold]
                    if Path(f"{FEATURES_RT}/{p}_frames.npy").exists()
                    and Path(f"{FEATURES_IT}/{p}_frames.npy").exists()
                ]
                for i, pid in enumerate(test_ids):
                    fold_rows.append({
                        "condition": condition,
                        "fold": fold,
                        "rep": rep,
                        "pid": pid,
                        "prediction": preds[i],
                        "target": targets[i],
                    })

                new_rows = pd.DataFrame(fold_rows)
                if pred_df.empty:
                    pred_df = new_rows
                else:
                    pred_df = pd.concat([pred_df, new_rows], ignore_index=True)
                pred_df.to_csv(pred_file, index=False)
                done_folds.add((condition, fold, rep))

                # Log per-fold result for monitoring
                fold_metrics = compute_metrics(preds, targets)
                logger.info(
                    f"  -> Fold {fold} Acc={fold_metrics['accuracy']:.1f}%  "
                    f"F1={fold_metrics['f1']:.1f}%  ({elapsed/60:.1f}min)"
                )

                all_preds_this_rep.extend(preds)
                all_targets_this_rep.extend(targets)

            except Exception as e:
                logger.error(f"  ERROR fold={fold} rep={rep}: {e}")
                break  # cannot pool partial rep, skip to next rep
        else:
            # All folds completed for this rep. Pool and compute metrics.
            pooled_metrics = compute_metrics(all_preds_this_rep, all_targets_this_rep)

            pooled_row = pd.DataFrame([{
                "condition": condition,
                "rep": rep,
                "n_participants": len(all_preds_this_rep),
                "accuracy": pooled_metrics["accuracy"],
                "precision": pooled_metrics["precision"],
                "recall": pooled_metrics["recall"],
                "f1": pooled_metrics["f1"],
                "elapsed_s": total_elapsed,
            }])
            if pooled_df.empty:
                pooled_df = pooled_row
            else:
                pooled_df = pd.concat([pooled_df, pooled_row], ignore_index=True)
            pooled_df.to_csv(pooled_file, index=False)
            done_reps.add((condition, rep))

            logger.info(
                f"  REP {rep} POOLED ({len(all_preds_this_rep)} participants): "
                f"Acc={pooled_metrics['accuracy']:.1f}%  "
                f"F1={pooled_metrics['f1']:.1f}%"
            )

    # Summary for this condition
    sub = pooled_df[pooled_df["condition"] == condition] if not pooled_df.empty else pd.DataFrame()
    if len(sub) > 0:
        total_time = (time.time() - t_cond) / 60
        print(f"\n{'=' * 60}")
        print(f"{condition} COMPLETE ({len(sub)}/{N_REPS} reps, {total_time:.0f}min total)")
        print(f"  Pooled Acc: {sub['accuracy'].mean():.1f} +/- {sub['accuracy'].std():.1f}%")
        print(f"  Pooled F1:  {sub['f1'].mean():.1f} +/- {sub['f1'].std():.1f}%")
        print(f"  Predictions saved to: {pred_file}")
        print(f"  Pooled results saved to: {pooled_file}")
        print(f"{'=' * 60}\n")


# ===========================================================================
# Results summary
# ===========================================================================
def print_summary(k_folds):
    pooled_file = os.path.join(RESULTS_DIR, "pooled_results.csv")
    if not os.path.exists(pooled_file):
        print("No pooled results yet.")
        return

    df = pd.read_csv(pooled_file)
    total_reps = 7 * N_REPS
    print(f"\nPooled reps completed: {len(df)} / {total_reps}")
    print(f"(Each rep = k={k_folds} folds pooled over all ~110 participants)")
    print()
    print(f"{'Condition':<15} {'Acc':>8} {'+-':>5} {'F1':>8} {'+-':>5} {'N reps':>6}")
    print("-" * 55)
    order = [
        "ba1_rt", "ba1_it", "itmla_rt", "itmla_it",
        "ctga_rt", "ctga_it", "full_cdma",
    ]
    for cond in order:
        s = df[df["condition"] == cond]
        if len(s) == 0:
            print(f"{cond:<15} -- not run yet")
        else:
            print(
                f"{cond:<15}"
                f"{s['accuracy'].mean():>8.1f}"
                f"{s['accuracy'].std():>5.1f}"
                f"{s['f1'].mean():>8.1f}"
                f"{s['f1'].std():>5.1f}"
                f"{len(s):>6}"
            )
    print()
    print("Thesis targets (k=3):")
    print("  ba1_rt:    Acc=85.5+-1.2  F1=84.7+-1.3")
    print("  itmla_rt:  Acc=89.2+-0.7  F1=89.0+-0.6")
    print("  ctga_rt:   Acc=90.4+-1.2  F1=90.7+-1.0")
    print("  full_cdma: Acc=92.7+-0.8  F1=92.5+-0.8")
    print()
    print("Key check: ordering ba1 < itmla < ctga < full_cdma must hold.")


# ===========================================================================
# Git push results back to GitHub
# ===========================================================================
def push_results_to_github():
    """
    Commit and push results CSV (and any logs) back to the GitHub repo.
    Assumes the script is running from inside the cloned repo, or that
    RESULTS_DIR is inside the repo.
    """
    repo_root = find_git_root()
    if repo_root is None:
        logger.warning("Not inside a git repo. Skipping push.")
        return False

    pooled_file = os.path.join(RESULTS_DIR, "pooled_results.csv")
    results_rel = os.path.relpath(pooled_file, repo_root)
    results_dir_rel = os.path.relpath(RESULTS_DIR, repo_root)

    try:
        subprocess.run(
            ["git", "add", results_dir_rel],
            cwd=repo_root, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m",
             f"exp1: update results ({time.strftime('%Y-%m-%d %H:%M')})"],
            cwd=repo_root, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=repo_root, check=True, capture_output=True,
        )
        logger.info(f"Results pushed to GitHub ({results_rel})")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git push failed: {e.stderr.decode().strip()}")
        return False


def find_git_root():
    """Walk up from RESULTS_DIR to find the .git root."""
    path = Path(RESULTS_DIR).resolve()
    while path != path.parent:
        if (path / ".git").exists():
            return str(path)
        path = path.parent
    # Also check the script's own directory
    script_dir = Path(__file__).resolve().parent
    while script_dir != script_dir.parent:
        if (script_dir / ".git").exists():
            return str(script_dir)
        script_dir = script_dir.parent
    return None


# ===========================================================================
# Sanity check
# ===========================================================================
def verify_all_modes(device):
    """Quick forward pass through all 7 modes to catch shape bugs early."""
    dummy = {
        "rt_frames": torch.randn(2, 5, 128, 32).to(device),
        "it_frames": torch.randn(2, 8, 128, 32).to(device),
        "rt_mask": torch.ones(2, 5).to(device),
        "it_mask": torch.ones(2, 8).to(device),
        "n_rt": torch.tensor([5, 5]).to(device),
        "n_it": torch.tensor([8, 8]).to(device),
        "labels": torch.tensor([0, 1]).to(device),
    }
    for mode in CDMAModel.VALID_MODES:
        m = CDMAModel(mode).to(device)
        out = m(dummy)
        assert out["p_hat"].shape == (2, 1), f"{mode}: bad shape"
        assert not torch.isnan(out["p_hat"]).any(), f"{mode}: NaN output"
    logger.info("All 7 modes verified.")


# ===========================================================================
# Main
# ===========================================================================
def main():
    global DEVICE, FEATURES_RT, FEATURES_IT, FOLD_FILE, RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="CDMA Replication - Experiment 2 (pooled k-fold evaluation)",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help=(
            "Which conditions to run. Use 'all' for all 7, or list "
            "specific ones: ba1_rt ba1_it itmla_rt itmla_it ctga_rt ctga_it full_cdma. "
            "Required unless --test or --summary-only is used."
        ),
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory to download/extract data into (default: ./data)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Directory to store result CSVs (default: ./results)",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Push results to GitHub after all conditions complete",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Just print the results summary table and exit",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=(
            "Quick pipeline test: all 7 conditions, 10 epochs, 1 rep, "
            "separate results_test/ directory. Confirms the whole setup "
            "works before committing to the full run."
        ),
    )
    args = parser.parse_args()

    # --test mode: override settings for a quick pipeline check
    if args.test:
        global EPOCHS, N_REPS
        EPOCHS = 10
        N_REPS = 1
        args.results_dir = os.path.join(os.path.dirname(os.path.abspath(args.results_dir)), "results_test")
        args.conditions = ["all"]
        # Wipe previous test results for a clean run
        import shutil
        test_dir = os.path.abspath(args.results_dir)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        logger.info("=" * 60)
        logger.info("TEST MODE: 10 epochs, 1 rep, all 7 conditions")
        logger.info(f"Test results dir: {test_dir}")
        logger.info("This is NOT a real run. Just checking the pipeline.")
        logger.info("=" * 60)

    # Validate that --conditions is provided when needed
    if args.conditions is None and not args.test and not args.summary_only:
        parser.error("--conditions is required (use 'all' or list specific ones, or use --test)")

    # Set up paths
    RESULTS_DIR = os.path.abspath(args.results_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Set up device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {DEVICE}  |  GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Download and extract data
    corpus_root = download_and_extract_data(os.path.abspath(args.data_dir))
    FEATURES_RT = os.path.join(corpus_root, "cdma_features", "rt")
    FEATURES_IT = os.path.join(corpus_root, "cdma_features", "it")
    FOLD_FILE = os.path.join(corpus_root, "fold-lists.csv")

    # Verify data
    rt_count = len(list(Path(FEATURES_RT).glob("*_frames.npy")))
    it_count = len(list(Path(FEATURES_IT).glob("*_frames.npy")))
    assert rt_count == 110, f"Expected 110 RT frame files, found {rt_count}"
    assert it_count == 110, f"Expected 110 IT frame files, found {it_count}"
    logger.info(f"Data OK: RT={rt_count}, IT={it_count} frame files")

    # Build participant info and folds
    all_pids, label_map = build_participant_info()
    folds_data = parse_fold_lists()
    fold_to_pids = {
        f: [p for p in pids if p in set(all_pids)]
        for f, pids in folds_data["read"].items()
    }
    k_folds = len(fold_to_pids)

    n_ctrl = sum(1 for v in label_map.values() if v == 0)
    n_dep = sum(1 for v in label_map.values() if v == 1)
    logger.info(f"Participants: {len(all_pids)}  Control: {n_ctrl}  Depressed: {n_dep}")
    logger.info(f"Folds: {k_folds}")
    for f in range(1, k_folds + 1):
        pids = fold_to_pids[f]
        deps = sum(label_map.get(p, 0) for p in pids)
        logger.info(
            f"  Fold {f}: {len(pids)} participants, "
            f"{deps} depressed ({100 * deps // max(len(pids), 1)}%)"
        )

    # Summary only mode
    if args.summary_only:
        print_summary(k_folds)
        return

    # Verify model modes
    verify_all_modes(DEVICE)

    # Resolve which conditions to run
    all_modes = list(CDMAModel.VALID_MODES)
    if "all" in args.conditions:
        conditions = all_modes
    else:
        conditions = []
        for c in args.conditions:
            if c not in all_modes:
                parser.error(
                    f"Unknown condition '{c}'. "
                    f"Valid: {', '.join(all_modes)} or 'all'"
                )
            conditions.append(c)

    logger.info(f"Conditions to run: {conditions}")

    # Run each condition
    for condition in conditions:
        run_condition(condition, all_pids, label_map, fold_to_pids, k_folds)

    # Print final summary
    print_summary(k_folds)

    # Test mode verification
    if args.test:
        pooled_file = os.path.join(RESULTS_DIR, "pooled_results.csv")
        print("\n" + "=" * 60)
        print("TEST MODE VERIFICATION")
        print("=" * 60)
        if os.path.exists(pooled_file):
            df = pd.read_csv(pooled_file)
            all_modes = list(CDMAModel.VALID_MODES)
            conditions_done = sorted(df["condition"].unique())
            conditions_expected = sorted(all_modes)
            n_done = len(conditions_done)
            n_expected = len(conditions_expected)

            # Check all 7 conditions ran
            if conditions_done == conditions_expected:
                print(f"  [PASS] All {n_expected} conditions completed")
            else:
                missing = set(conditions_expected) - set(conditions_done)
                print(f"  [FAIL] {n_done}/{n_expected} conditions. Missing: {missing}")

            # Check each has 1 rep with correct participant count
            all_ok = True
            for cond in conditions_expected:
                sub = df[df["condition"] == cond]
                if len(sub) == 0:
                    print(f"  [FAIL] {cond}: no results")
                    all_ok = False
                    continue
                n_part = sub["n_participants"].iloc[0]
                acc = sub["accuracy"].iloc[0]
                f1 = sub["f1"].iloc[0]
                print(f"  [{'PASS' if n_part == len(all_pids) else 'FAIL'}] {cond}: "
                      f"{n_part} participants pooled, Acc={acc:.1f}%, F1={f1:.1f}%")
                if n_part != len(all_pids):
                    all_ok = False

            # Check ablation ordering (on 10 epochs it may not hold, just report)
            print()
            print("  Ablation ordering (may not hold at 10 epochs, just for info):")
            order_pairs = [
                ("ba1_rt", "itmla_rt"), ("itmla_rt", "ctga_rt"),
                ("ctga_rt", "full_cdma"),
            ]
            for a, b in order_pairs:
                sa = df[df["condition"] == a]["f1"].values
                sb = df[df["condition"] == b]["f1"].values
                if len(sa) > 0 and len(sb) > 0:
                    holds = sb[0] >= sa[0]
                    print(f"    {a} ({sa[0]:.1f}) -> {b} ({sb[0]:.1f}): "
                          f"{'holds' if holds else 'does NOT hold (expected at 10 epochs)'}")

            print()
            if all_ok:
                print("  PIPELINE TEST PASSED. Safe to run the full experiment.")
                print("  Command for full run:")
                print("    python cdma_experiment.py --conditions all")
            else:
                print("  PIPELINE TEST FAILED. Check errors above.")
        else:
            print("  [FAIL] No pooled_results.csv produced.")
        print("=" * 60)
        return

    # Push to GitHub if requested
    if args.push:
        push_results_to_github()


if __name__ == "__main__":
    main()
