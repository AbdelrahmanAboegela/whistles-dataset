---
license: mit
---
# Volleylitics Whistle Detection Dataset


[HuggingFace dataset link](https://huggingface.co/datasets/GYdevy/volleyball-whistles)

## Overview

This dataset contains full match audio recordings and whistle annotations
for training and evaluating whistle detection models in indoor volleyball environments.

The dataset is designed for:

- DSP-based whistle detection
- Audio classification (CNN / CRNN)
- Acoustic event detection research

---

## Audio Format

- Format: WAV
- Sample rate: 22.05kHz 

---

## Structure

```
match1.wav 
match2.wav  
...  

whistles_all.json            # Raw manual labels
whistles_all_anchored.json   # Band-peak anchored labels
whistles_all_reanchored.json # Re-anchored ground truth (recommended for evaluation)

dsp_detector.py              # DSP baseline detector (Stage A–C, no ML)
ml_classifier.py             # ML second-stage classifier (WhistleCNN / WhistleCRNN)
```

---

## Annotation Format

Each annotation JSON contains:

```json
{
    "match_id": "match1",
    "whistle_id": 0,
    "time": 395.705,
    "type": "other",
    "t_raw": 395.705,
    "global_id": 0,
    "t_anchor": 395.663
},
{
    "match_id": "match1",
    "whistle_id": 1,
    "time": 396.473,
    "type": "other",
    "t_raw": 396.473,
    "global_id": 1,
    "t_anchor": 396.303
},
```

`t_raw` is the time of a manual label pointing to a whistle sound.  
`t_anchor` is the time refined by spectral peak features (band_peak) and is the
recommended ground truth for evaluation.

Whistle type labels:

| Type | Description |
|------|-------------|
| `serve` | Serve whistle |
| `rally_end` | Rally-end whistle |
| `other` | Timeouts, set end, side switch, subs, administrative double whistles |

---

## DSP Baseline (`dsp_detector.py`)

The included detector is a **pure DSP + rule-based pipeline** (no ML).

### Pipeline

1. **Stage A — ROI detector**: STFT over the 2.5–6 kHz band; combines band
   energy, sharpness, spectral flatness, and spectral flux into a per-frame
   score; hysteresis thresholding selects active frames.
2. **Grouping**: Active frames are merged into contiguous groups (max gap = 12
   frames ≈ 70 ms).
3. **Candidate extraction**: Groups longer than `min_frames` (≈ 87 ms) are
   expanded with a 350 ms pad on each side.
4. **Refinement**: Each candidate is re-centered using a combination of band
   peak energy and spectral flux onset detection within the narrow 3.7–4.3 kHz
   band.
5. **Feature extraction**: Eight discriminative features are computed per
   candidate window (band ratio, frequency stability, band flatness, peak
   prominence, band energy, narrow-band concentration, spectral flux, tonal
   ratio).
6. **Rule-based sifter**: Candidates are scored using z-score–normalised
   features; low-scoring detections are rejected to reduce the explosion ratio.

### Baseline Performance (multi-match evaluation)

| Metric | Value |
|--------|-------|
| Frame Recall | ~0.99 |
| Group Recall | ~0.99 |
| Candidate Recall | ~0.95–0.96 |
| Explosion Ratio | ~2.5 |
| Median centering offset | ~70 ms |
| 90th-percentile offset | ~750 ms |

### Running the detector

Install dependencies:

```bash
pip install librosa soundfile scipy numpy
```

Edit the path constants at the top of `dsp_detector.py` to point to your local
video and ground-truth files, then run:

```bash
python dsp_detector.py
```

---

## ML Classifier (`ml_classifier.py`)

### Why DSP first?

The DSP baseline was built first because it:

- requires **zero labelled training data** (no GPU, no model download)
- achieves **≥ 0.99 recall** immediately, providing a reliable high-recall
  candidate generator
- produces the **positive/negative WAV snippets** (`pos/` / `neg/`) that
  become the training data for the ML model

### Hybrid pipeline

```
Full match audio
      │
      ▼
 DSP detector          ← dsp_detector.py  (high recall, ~2.5× explosion)
      │  candidate windows
      ▼
 ML second-stage       ← ml_classifier.py  (reduces explosion ratio)
      │  accepted detections
      ▼
 Final output
```

### Available architectures

| Class | Description |
|-------|-------------|
| `WhistleCNN` | Compact 2-D CNN on log-mel spectrograms (default, fast) |
| `WhistleCRNN` | CNN + bidirectional GRU — models onset/decay dynamics |

### Install

```bash
pip install torch torchaudio librosa
```

### Train

```bash
# generate pos/ neg/ snippets first:
python dsp_detector.py

# train CNN (default)
python ml_classifier.py train \
    --pos path/to/pos \
    --neg path/to/neg \
    --out whistle_cnn.pt

# train CRNN variant
python ml_classifier.py train \
    --pos   path/to/pos \
    --neg   path/to/neg \
    --model crnn \
    --out   whistle_crnn.pt
```

### Evaluate

```bash
python ml_classifier.py eval \
    --pos   path/to/pos \
    --neg   path/to/neg \
    --model whistle_cnn.pt
```

### Activate the ML sifter in dsp_detector.py

Set `ML_MODEL_PATH` at the top of `dsp_detector.py` to the checkpoint:

```python
ML_MODEL_PATH = r"path/to/whistle_cnn.pt"
ML_THRESHOLD  = 0.45   # lower → higher recall
```

---

## Encoder Exploration

The `ml_classifier.py` CNN/CRNN is trained from scratch.  Swapping the
encoder for a pre-trained audio representation can significantly improve
accuracy when labelled data is limited.

| Encoder | Description | Install |
|---------|-------------|---------|
| **WhistleCNN / WhistleCRNN** | Trained from scratch on dataset snippets | *(included)* |
| **PANNs** | Pretrained Audio Neural Networks (AudioSet-scale, 2048-d embeddings) | `pip install panns-inference` |
| **CLAP** | Contrastive Language–Audio Pre-training — zero-shot capable ("sound of a whistle") | `pip install msclap` |
| **wav2vec 2.0 / HuBERT** | Frame-level self-supervised speech features, strong for temporal modelling | `pip install transformers` |
| **EnCodec** | Meta neural audio codec — quantised residual embeddings, strong for short clips | `pip install encodec` |

#### Using PANNs
```python
from panns_inference import AudioTagging
at  = AudioTagging(checkpoint_path=None)
emb = at.inference(audio)['embedding']   # 2048-d; attach a linear head
```

#### Using CLAP (zero-shot)
```python
from msclap import CLAP
clap  = CLAP(version='2023', use_cuda=False)
texts = ["sound of a referee whistle", "crowd noise"]
scores = clap.get_similarity(audio_paths, texts)
```

#### Using wav2vec 2.0
```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor
model  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
proc   = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
hidden = model(**inputs).last_hidden_state.mean(dim=1)  # (B, 768)
# attach a linear classification head
```

---

## Open Challenge

Can you beat or improve the DSP baseline?  Goals:

- Maintain **≥ 0.99 recall** while reducing explosion ratio below 2.0
- Improve whistle centering accuracy (lower 90th-percentile offset)
- Increase robustness under heavy crowd noise

### Possible Directions

- Band-restricted modeling (2.5–6 kHz focus)
- Spectral flux onset detection for better centering
- CNN / CRNN architectures trained on the positive/negative snippets
- Pre-trained audio encoders (PANNs, CLAP, wav2vec 2.0, EnCodec)
- Hybrid DSP + ML pipelines
- Sequence models (HMM / temporal smoothing) over DSP candidates
- Tonal purity / harmonic ratio features

Creative approaches and discussions are welcome — open an issue or PR!
