# Privacy-Preserving-Audio-Deep-Fake-Detection
# Variable Bitrate Residual Vector Quantization for Audio Coding

Python 3.9.13
argbind>=0.3.7
descript-audiotools>=0.7.2
einops
numpy==1.26
tqdm
numba>=0.5.7
tensorboard
torchmetrics

## Overview

This project explores the use of **Variable Bitrate Residual Vector Quantization (VRVQ)** for audio coding and its integration with the **SafeEar** pipeline for deepfake audio detection.

The work evaluates:

* Impact of varying audio bitrates on VRVQ performance
* A combined **VRVQ + SafeEar pipeline** for spoof detection
* Trade-offs between accuracy, latency, and robustness

---

## Part 1: VRVQ Audio Coding

### Implementation

* The model requires `.wav` input files.
* Original dataset was in `.flac` format.
* Converted `.flac → .wav` using:

  * `librosa`
  * `soundfile`
* Updated configuration files to:

  * Point to the dataset
  * Optimize execution for HPC

### Training

* Trained VRVQ model on processed dataset
* Generated trained weights: `weights.pth`

### Bitrate Experiments

* Selected 10 audio samples
* Modified sample rates (treated as bitrate variation):

  * 8000 Hz
  * 16000 Hz
  * 22050 Hz
  * 44100 Hz

### Observation

* Inference time increases **logarithmically** with increasing bitrate
* Higher bitrate → more data → higher computational cost

---

## Part 2: New Pipeline — VRVQ + SafeEar

### Pipeline Architecture

```
Audio → VRVQ → Frontend CDM → HuBERT → SafeEar
```

---

## Implementation

### Dataset Preparation

1. Processed entire **ASVspoof dataset** through VRVQ
2. Converted outputs:

   * `.wav → .flac`
   * Downsampled to **16 kHz**

### Feature Extraction

* Used **HuBERT model**
* Generated token files (`.npy`) for each audio sample

### Model Training

* Trained **SafeEar** using generated tokens

### Evaluation

* Built test pipeline to compute:

  * EER (Equal Error Rate)
  * min t-DCF
  * Average Latency
  * Throughput

---

## Results

### VRVQ + SafeEar

* **EER:** 12.64%
* **min t-DCF:** 0.0070
* **Avg Latency:** 15.57 ms/sample
* **Throughput:** 64.22 samples/sec

### Baseline (SafeEar)

* **EER:** 4.43%
* **min t-DCF:** 0.05
* **Avg Latency:** 4.22 ms/sample
* **Throughput:** 236.68 samples/sec

---

## Analysis

### Performance Trade-offs

| Metric     | VRVQ + SafeEar | SafeEar | Observation               |
| ---------- | -------------- | ------- | ------------------------- |
| EER        | Higher         | Lower   | Worse user discrimination |
| min t-DCF  | Lower          | Higher  | Better spoof detection    |
| Latency    | Higher         | Lower   | Slower                    |
| Throughput | Lower          | Higher  | Less efficient            |

---

### Key Insight

* **Higher EER** → worse at distinguishing real users
* **Lower t-DCF** → better at detecting spoof attacks

👉 This indicates:

* VRVQ improves **security robustness**
* But reduces **classification accuracy**

---

## Why VRVQ Helps

VRVQ:

* Blurs background noise
* Suppresses silent/adversarial regions
* Removes subtle perturbations used in attacks

👉 Result:

* Adversarial signals are weakened
* Spoof detection improves

---

## Important Observation on Bitrate

Example (single audio sample):

| Stage               | Bitrate      |
| ------------------- | ------------ |
| Original `.flac`    | 133.35 kbps  |
| Converted `.wav`    | 1411.36 kbps |
| After VRVQ (`.wav`) | 705.69 kbps  |
| Back to `.flac`     | 125.63 kbps  |

### Insight

* VRVQ compresses **intermediate waveform representation**
* Final `.flac` bitrate remains similar to original
* But **signal characteristics are altered**

---

## Conclusion

* VRVQ does **not significantly reduce final storage bitrate**
* Instead, it:

  * Removes adversarial artifacts
  * Improves spoof robustness
  * Degrades fine-grained user distinction

### Final Takeaway

> VRVQ acts as a **defensive preprocessing layer**, not a compression tool.

---

## Notes

* VRVQ pipeline involves conversion:

  * `.flac → .wav → VRVQ → .wav → .flac`
* Experiments conducted on HPC environment

---

