# Evaluation Report: AI-Based Fraudulent Phone Call Detection System

**Document Version:** 1.0.0  
**Generated On:** [YYYY-MM-DD HH:MM:SS UTC]  
**Model Identifier:** [BiLSTM-TextClassifier-v1.0]  
**Evaluation Scope:** [Held-Out Test Set / Cross-Validation / Exploratory]  
**Status:** [Validated / Preliminary]

---

## 1. Executive Summary

This evaluation report details the performance of the Bidirectional Long Short-Term Memory (BiLSTM) phone-call fraud detection architecture. The system processes phone call audio via OpenAI Whisper speech-to-text, extracts fraud-relevant semantic indicators, and predicts the probability that an incoming phone call is fraudulent.

### Key Results Snapshot

| Metric | Score | Target Threshold | Status |
|---|---|---|---|
| **Accuracy** | [0.0000] | > 0.8500 | [MET / UNMET] |
| **Fraud Recall (Sensitivity)** | [0.0000] | > 0.9000 | [MET / UNMET] |
| **Fraud Precision** | [0.0000] | > 0.8000 | [MET / UNMET] |
| **F1 Score** | [0.0000] | > 0.8500 | [MET / UNMET] |
| **ROC-AUC** | [0.0000] | > 0.9000 | [MET / UNMET] |
| **PR-AUC** | [0.0000] | > 0.8500 | [MET / UNMET] |
| **False Positive Rate (FPR)** | [0.0000] | < 0.1000 | [MET / UNMET] |
| **False Negative Rate (FNR)** | [0.0000] | < 0.1000 | [MET / UNMET] |

> **Critical Notice on Metric Interpretation:**  
> In telephony fraud detection, **Fraud Recall** is prioritized to prevent dangerous scams (e.g. IRS impersonation, bank account draining) from reaching end users. However, False Positive Rate (FPR) is strictly monitored to prevent legitimate business and personal communications from being incorrectly blocked.

---

## 2. Dataset Provenance & Integrity

### Primary Source: NCSU Robocall Audio Dataset
- **Repository:** `https://github.com/wspr-ncsu/robocall-audio-dataset`
- **Origin:** Real-world robocall audio recordings originating from FTC "Project Point of No Entry" (PPoNE) enforcement actions.
- **Evidence Backing:** Official FTC cease-and-desist letters and warning letters provided in `pdf_files/`.
- **Label Status:** All NCSU records are verified fraud (`label = fraud`, `label_status = verified`).

### Negative Class (Legitimate) Requirement
- The primary NCSU dataset contains exclusively fraudulent robocalls.
- Supervised binary training requires legitimate call recordings (e.g., from Common Voice or customer service dialogues) to establish the negative class.
- Records with `label_status = unknown` are strictly excluded from supervised training and retained only for exploratory clustering.
- Data fabrication is strictly prohibited.

### Dataset Split Distribution

| Split | Total Samples | Verified Fraud | Verified Legitimate | Weak / Exploratory | Group Count (Sources) |
|---|---|---|---|---|---|
| **Training** | [0] | [0] | [0] | [0] | [0] |
| **Validation** | [0] | [0] | [0] | [0] | [0] |
| **Held-Out Test**| [0] | [0] | [0] | [0] | [0] |
| **Total** | [0] | [0] | [0] | [0] | [0] |

---

## 3. Model Architecture & Hyperparameters

```
Input Transcript (string)
  │
  ▼
TextVectorization Layer (Vocabulary: [20,000], Sequence Length: [256])
  │
  ▼
Embedding Layer (Dimension: [128])
  │
  ▼
Bidirectional LSTM (Units: [128], Dropout: [0.3])
  │
  ▼
Dense Layer (Units: [64], Activation: ReLU)
  │
  ▼
Dropout (Rate: [0.15])
  │
  ▼
Output Layer (Units: 1, Activation: Sigmoid)
```

- **Optimizer:** Adam (Initial Learning Rate: [0.001])
- **Loss Function:** Binary Cross-Entropy
- **Callbacks:** EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=3), ModelCheckpoint (monitor=val_recall)
- **Decision Threshold:** [0.50] (calibrated on validation set for optimal recall-precision trade-off)

---

## 4. Evaluation Metrics Breakdown

### 4.1 Confusion Matrix

| Actual \ Predicted | Predicted Legitimate (0) | Predicted Fraud (1) | Total |
|---|---|---|---|
| **Actual Legitimate (0)** | **TN:** [0] | **FP:** [0] | [0] |
| **Actual Fraud (1)** | **FN:** [0] | **TP:** [0] | [0] |
| **Total** | [0] | [0] | [0] |

### 4.2 Error Analysis

- **False Positives (Legitimate flagged as Fraud):**
  - Count: [0]
  - Primary contributing factors: [e.g., legitimate security notifications containing words like "verify", "account", or "alert"].
- **False Negatives (Fraud flagged as Legitimate):**
  - Count: [0]
  - Primary contributing factors: [e.g., conversational scams without overt threat keywords, novel phishing scripts, short audio snippets].

---

## 5. Segment-Level vs. Full-Call Analysis

| Analysis Mode | Fraud Recall | Precision | Latency per Call (s) | Annotation Quality |
|---|---|---|---|---|
| **Full-Call Transcript** | [0.0000] | [0.0000] | [0.00] | Verified / Call-Level |
| **Segment-Level (30s / 15s stride)** | [0.0000] | [0.0000] | [0.00] | Exploratory (Weakly Supervised) |

> **Segment Provenance Note:**  
> Segment-level predictions are marked exploratory unless human segment-level ground truth is provided. Full-call labels cannot be blindly inherited by individual segments without risking label noise.

---

## 6. Language & Multilingual Generalization

| Spoken Language | Total Evaluated | Whisper Detection Acc (%) | Classification F1 |
|---|---|---|---|
| **English (en)** | [0] | [0.0%] | [0.0000] |
| **Mandarin (zh)** | [0] | [0.0%] | [0.0000] |
| **Hindi / Hinglish (hi/en)** | [0] | [0.0%] | [0.0000] |

---

## 7. Operational Recommendations

1. **Human-in-the-Loop Threshold:** All calls with calibrated fraud probability in the range `[0.45, 0.75]` must be routed to human analysts with status `"needs_review = true"`.
2. **Audio Preprocessing:** Ensure 16 kHz mono resampling is applied uniformly before Whisper transcription to avoid acoustic distortion.
3. **Continuous Evaluation:** Regularly evaluate on new FTC cease-and-desist filings to monitor scam script drift.

---

## 8. Disclaimer & Research Limitations

> **LEGAL AND OPERATIONAL DISCLAIMER:**  
> The metrics presented in this report represent statistical model evaluations under controlled experimental conditions. **Predictions generated by this system are probabilistic estimates, not legal or evidentiary proof of fraud or criminal activity.** This system is designed as an investigative assistance tool and should not be used as the sole basis for punitive, financial, or legal actions without independent human verification.
