"""
Hybrid DSP + ML Whistle Classifier
====================================
Second-stage ML classifier that operates on the 0.6-second candidate
snippets produced by the DSP detector (dsp_detector.py).

Why DSP first?
--------------
The DSP pipeline runs with no labelled data and achieves ≥ 0.99 recall,
making it an excellent high-recall candidate generator.  The ML model is
then trained only on those candidates, drastically reducing the search
space and the amount of labelled audio needed.

Model architectures
-------------------
WhistleCNN  : lightweight 2-D CNN on log-mel spectrograms (default)
WhistleCRNN : CNN + bidirectional GRU — stronger temporal context, useful
              when the whistle onset/offset matters for classification

Encoder options explored
------------------------
The log-mel CNN above is trained from scratch.  The alternatives below
swap the encoder for a richer pre-trained audio representation:

1. **WhistleCNN / WhistleCRNN** (this file)
   - Trained from scratch on the dataset snippets.
   - Pros: tiny, fast, no external model downloads.
   - Cons: limited by the size of the training set.

2. **PANNs** (Pretrained Audio Neural Networks, Kong et al. 2020)
   - pip install panns-inference
   - 2048-d embeddings from AudioSet-scale training.
   - Use: `from panns_inference import AudioTagging; at = AudioTagging(checkpoint_path=None)`
   - Replace the CNN encoder with `at.inference(audio)['embedding']` and attach a linear head.

3. **CLAP** (Contrastive Language–Audio Pre-training, LAION 2023)
   - pip install msclap
   - 512-d embeddings; zero-shot capable ("sound of a whistle").
   - Use: `from msclap import CLAP; clap = CLAP(version='2023', use_cuda=False)`
   - Embed with `clap.get_audio_embeddings([path])` or fine-tune the projection head.

4. **wav2vec 2.0 / HuBERT** (Meta / HuggingFace)
   - pip install transformers
   - Frame-level features, excellent for temporal modelling.
   - Use: `Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")` + mean-pool + linear head.

5. **EnCodec** (Meta neural audio codec)
   - pip install encodec
   - Quantised residual embeddings; strong for short audio segments.
   - Use: `EncodecModel.encodec_model_24khz()` + decode codebook indices as features.

Usage
-----
# 1. Generate snippets with dsp_detector.py first, then:

# Train
python ml_classifier.py train \\
    --pos  path/to/pos \\
    --neg  path/to/neg \\
    --out  whistle_cnn.pt

# Train CRNN variant
python ml_classifier.py train \\
    --pos  path/to/pos \\
    --neg  path/to/neg \\
    --model crnn \\
    --out  whistle_crnn.pt

# Evaluate snippet-level metrics
python ml_classifier.py eval \\
    --pos   path/to/pos \\
    --neg   path/to/neg \\
    --model whistle_cnn.pt

# Integration with dsp_detector.py (see evaluate_match there):
#   set ML_MODEL_PATH at the top of dsp_detector.py to use this as
#   a second-stage sifter after rule_based_sifter.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import librosa


# ============================================================
# AUDIO / FEATURE CONFIG
# ============================================================

SR          = 22050
SNIPPET_SEC = 0.6       # must match dsp_detector.py SNIPPET_SEC
N_MELS      = 64
N_FFT       = 1024
HOP         = 128
FMIN        = 2500.0    # focus on referee whistle frequency range
FMAX        = 8000.0
SEED        = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def wav_to_logmel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Convert a mono waveform to a normalised log-mel spectrogram (n_mels × T)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class WhistleDataset(Dataset):
    """
    Loads positive (whistle) and negative (non-whistle) WAV snippets
    produced by dsp_detector.generate_dataset_snippets().

    Parameters
    ----------
    pos_dir  : directory containing whistle WAV snippets
    neg_dir  : directory containing non-whistle WAV snippets
    augment  : apply lightweight data augmentation during training
    """

    def __init__(self, pos_dir: str, neg_dir: str, augment: bool = False):
        self.samples: list = []
        self.augment = augment

        for name in sorted(os.listdir(pos_dir)):
            if name.endswith(".wav"):
                self.samples.append((os.path.join(pos_dir, name), 1))

        for name in sorted(os.listdir(neg_dir)):
            if name.endswith(".wav"):
                self.samples.append((os.path.join(neg_dir, name), 0))

        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y, _ = librosa.load(path, sr=SR, mono=True)

        # Pad or trim to exact snippet length
        target_len = int(SNIPPET_SEC * SR)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        if self.augment:
            y = self._augment(y)

        feat = wav_to_logmel(y)          # (n_mels, T)
        feat = feat[np.newaxis, ...]     # (1, n_mels, T)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def _augment(y: np.ndarray) -> np.ndarray:
        """Lightweight augmentations: random gain + additive Gaussian noise."""
        gain      = np.random.uniform(0.8, 1.2)
        noise_amp = np.random.uniform(0.0, 0.005)
        y = y * gain + noise_amp * np.random.randn(len(y)).astype(np.float32)
        return y.astype(np.float32)


# ============================================================
# MODEL — ConvBlock (shared)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: tuple = (3, 3), pool: tuple = (2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


# ============================================================
# MODEL — WhistleCNN
# ============================================================

class WhistleCNN(nn.Module):
    """
    Compact CNN whistle classifier trained from scratch on log-mel
    spectrograms.

    Input : (B, 1, N_MELS, T)  — log-mel spectrogram
    Output: (B, 2)             — logits [non-whistle, whistle]
    """

    def __init__(self, n_mels: int = N_MELS, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1,   32, pool=(2, 2)),   # → (32, n_mels/2, T/2)
            ConvBlock(32,  64, pool=(2, 2)),   # → (64, n_mels/4, T/4)
            ConvBlock(64, 128, pool=(2, 2)),   # → (128, n_mels/8, T/8)
        )
        self.gap     = nn.AdaptiveAvgPool2d(1)  # → (128, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(128, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)

    def embed(self, x) -> torch.Tensor:
        """Return 128-d embedding (useful for downstream tasks / visualisation)."""
        x = self.encoder(x)
        return self.gap(x).flatten(1)


# ============================================================
# MODEL — WhistleCRNN
# ============================================================

class WhistleCRNN(nn.Module):
    """
    CNN + bidirectional GRU for temporal context.

    The CNN front-end pools only along the frequency axis so the time
    dimension is preserved for the GRU to model temporal dynamics (onset,
    sustain, decay) — important for distinguishing whistles from shoe
    squeaks that have similar spectral content but different durations.

    Input : (B, 1, N_MELS, T)
    Output: (B, 2)
    """

    def __init__(self, n_mels: int = N_MELS,
                 gru_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1,  32, pool=(2, 1)),   # freq↓×2, time intact
            ConvBlock(32, 64, pool=(2, 1)),   # freq↓×2
            ConvBlock(64, 64, pool=(2, 1)),   # freq↓×2
        )
        freq_out     = n_mels // 8           # after 3× freq-pool-2
        self.gru     = nn.GRU(
            input_size  = 64 * freq_out,
            hidden_size = gru_hidden,
            num_layers  = 2,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(gru_hidden * 2, 2)

    def forward(self, x):
        x = self.cnn(x)                              # (B, 64, F', T)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T, C*F')
        _, h = self.gru(x)                            # h: (layers*2, B, H)
        h = torch.cat([h[-1], h[-2]], dim=-1)         # (B, 2H) last layer fwd+bwd
        return self.head(self.dropout(h))


# ============================================================
# TRAINING HELPERS
# ============================================================

def build_weighted_sampler(dataset: WhistleDataset) -> WeightedRandomSampler:
    """Over-sample the minority class to counteract class imbalance."""
    labels          = [s[1] for s in dataset.samples]
    counts          = np.bincount(labels)
    weights_per_cls = 1.0 / (counts + 1e-8)
    sample_weights  = [weights_per_cls[l] for l in labels]
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ============================================================
# TRAINING
# ============================================================

def train(pos_dir: str, neg_dir: str, out_path: str,
          model_type: str = "cnn",
          epochs: int = 30, batch_size: int = 32, lr: float = 1e-3,
          val_split: float = 0.15, patience: int = 7) -> None:
    """Train a WhistleCNN or WhistleCRNN and save the best checkpoint."""

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  model={model_type}")

    full_ds = WhistleDataset(pos_dir, neg_dir, augment=True)
    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    sampler      = build_weighted_sampler(train_ds.dataset)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = WhistleCRNN().to(device) if model_type == "crnn" \
            else WhistleCNN().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, epochs + 1):

        # ---- train ----
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # ---- validate ----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y   = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(x)
                correct  += (logits.argmax(1) == y).sum().item()
                total    += len(y)
        val_loss /= n_val
        val_acc   = correct / total

        scheduler.step()
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save({"model_type": model_type, "state_dict": model.state_dict()},
                       out_path)
            print(f"  → checkpoint saved to {out_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")


# ============================================================
# INFERENCE — second-stage ML sifter
# ============================================================

def load_model(checkpoint_path: str,
               device: torch.device) -> nn.Module:
    """Load a trained WhistleCNN or WhistleCRNN from a checkpoint file."""
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model = WhistleCRNN() if ckpt.get("model_type") == "crnn" \
            else WhistleCNN()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(device)


def score_candidates(y: np.ndarray, detections: list,
                     model: nn.Module, device: torch.device,
                     threshold: float = 0.5) -> list:
    """
    Second-stage ML sifter — drop candidates the model classifies as
    non-whistle.

    Parameters
    ----------
    y          : full match waveform (mono float32, sr=22050)
    detections : list of (start_sec, end_sec, t_peak) from DSP pipeline
    model      : trained WhistleCNN or WhistleCRNN
    device     : torch device
    threshold  : whistle probability threshold (lower → higher recall,
                 higher → lower explosion ratio)

    Returns
    -------
    accepted : filtered list of (start_sec, end_sec, t_peak)
    """
    target_len = int(SNIPPET_SEC * SR)
    half       = SNIPPET_SEC / 2
    accepted   = []

    for start, end, t_peak in detections:
        s0 = int((t_peak - half) * SR)
        s1 = s0 + target_len

        if s0 < 0 or s1 > len(y):
            # keep edge candidates unchanged to protect recall
            accepted.append((start, end, t_peak))
            continue

        snippet = y[s0:s1].astype(np.float32)
        feat    = wav_to_logmel(snippet)   # (n_mels, T)
        x       = torch.from_numpy(feat[np.newaxis, np.newaxis]).to(device)

        with torch.no_grad():
            prob = F.softmax(model(x), dim=1)[0, 1].item()  # P(whistle)

        if prob >= threshold:
            accepted.append((start, end, t_peak))

    return accepted


# ============================================================
# SNIPPET-LEVEL EVALUATION
# ============================================================

def evaluate(pos_dir: str, neg_dir: str, checkpoint_path: str,
             threshold: float = 0.5) -> None:
    """Report precision / recall / F1 on the snippet dataset."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(checkpoint_path, device)
    dataset = WhistleDataset(pos_dir, neg_dir, augment=False)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)[:, 1]
            preds = (probs >= threshold).long()
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n{'=' * 40}")
    print(f"Threshold : {threshold}")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1        : {f1:.3f}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whistle ML classifier")
    sub    = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train", help="Train a new classifier")
    p_train.add_argument("--pos",      required=True,  help="Directory of positive WAV snippets")
    p_train.add_argument("--neg",      required=True,  help="Directory of negative WAV snippets")
    p_train.add_argument("--out",      default="whistle_cnn.pt", help="Output checkpoint path")
    p_train.add_argument("--model",    default="cnn", choices=["cnn", "crnn"],
                         help="Model architecture")
    p_train.add_argument("--epochs",   type=int,   default=30)
    p_train.add_argument("--batch",    type=int,   default=32)
    p_train.add_argument("--lr",       type=float, default=1e-3)
    p_train.add_argument("--patience", type=int,   default=7,
                         help="Early-stopping patience (epochs)")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a trained classifier on snippets")
    p_eval.add_argument("--pos",       required=True, help="Directory of positive WAV snippets")
    p_eval.add_argument("--neg",       required=True, help="Directory of negative WAV snippets")
    p_eval.add_argument("--model",     required=True, dest="checkpoint",
                        help="Path to checkpoint (.pt)")
    p_eval.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for whistle probability")

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            args.pos, args.neg, args.out,
            model_type=args.model,
            epochs=args.epochs, batch_size=args.batch,
            lr=args.lr, patience=args.patience,
        )
    elif args.cmd == "eval":
        evaluate(args.pos, args.neg, args.checkpoint, args.threshold)
    else:
        parser.print_help()
