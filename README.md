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

dsp_detector.py              # DSP baseline detector
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

## Open Challenge

Can you beat or improve the DSP baseline?  Goals:

- Maintain **≥ 0.99 recall** while reducing explosion ratio below 2.0
- Improve whistle centering accuracy (lower 90th-percentile offset)
- Increase robustness under heavy crowd noise

### Possible Directions

- Band-restricted modeling (2.5–6 kHz focus)
- Spectral flux onset detection for better centering
- CNN / CRNN architectures trained on the positive/negative snippets
- Self-supervised audio embeddings (e.g. wav2vec, CLAP)
- Hybrid DSP + ML pipelines
- Sequence models (HMM / temporal smoothing) over DSP candidates
- Tonal purity / harmonic ratio features

Creative approaches and discussions are welcome — open an issue or PR!
