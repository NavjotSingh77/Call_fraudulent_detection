# Notebook 01: Exploratory Data Analysis & Manifest Inspection

This guide provides step-by-step instructions and Python workflows for exploring the **NCSU Robocall Audio Dataset** and the generated manifest.

---

## 1. Environment Setup & Imports

```python
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Insert project root
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.features.text_preprocessing import extract_fraud_features

config = load_config()
print("Config loaded successfully:", config["project"]["name"])
```

---

## 2. Load Dataset Manifest

```python
manifest_path = Path("data/metadata/manifest.csv")
if not manifest_path.exists():
    raise FileNotFoundError("Run 'python main.py download-data' first.")

df = pd.read_csv(manifest_path)
print(f"Total records in manifest: {len(df)}")
df.head(5)
```

Expected schema:
- `call_id`: Unique identifier (derived from audio filename).
- `audio_path`: Local filesystem path to processed or raw WAV file.
- `transcript`: Ground-truth or Whisper transcript.
- `language`: Spoken language code (e.g. `en`, `zh`, `hi`).
- `label`: Label assignment (`fraud`, `legitimate`, `uncertain`).
- `label_status`: Quality tier (`verified`, `weak`, `unknown`).
- `source`: Dataset source (e.g. `ncsu_ftc`).
- `recording_type`: Audio type (e.g. `robocall`).
- `case_pdf`: Associated FTC legal action document.

---

## 3. Label & Evidence Analysis

```python
print("Label Status Breakdown:")
print(df["label_status"].value_counts())

print("\nLabel Class Distribution:")
print(df["label"].value_counts())

# Note on NCSU Provenance:
# All NCSU recordings derive from FTC cease-and-desist actions.
# Therefore, verified robocalls are labeled 'fraud' with status 'verified'.
```

---

## 4. Linguistic & Language Breakdown

```python
plt.figure(figsize=(8, 4))
df["language"].value_counts().plot(kind="bar", color="steelblue")
plt.title("Language Distribution in Robocall Dataset")
plt.xlabel("Language Code")
plt.ylabel("Call Count")
plt.tight_layout()
plt.show()
```

---

## 5. Keyword & Risk Feature Analysis

```python
sample_transcripts = df["transcript"].dropna().sample(min(20, len(df)), random_state=42)
features_list = [extract_fraud_features(t) for t in sample_transcripts]

urgency_count = sum(1 for f in features_list if f.contains_urgency_terms)
threat_count = sum(1 for f in features_list if f.contains_threat_terms)
otp_count = sum(1 for f in features_list if f.contains_otp_terms)
financial_count = sum(1 for f in features_list if f.contains_financial_terms)

print(f"Urgency terms present in:    {urgency_count}/{len(features_list)} samples")
print(f"Threat terms present in:     {threat_count}/{len(features_list)} samples")
print(f"OTP terms present in:        {otp_count}/{len(features_list)} samples")
print(f"Financial terms present in:  {financial_count}/{len(features_list)} samples")
```

---

## 6. Audio Sample Inspection

```python
import soundfile as sf
import IPython.display as ipd

# Select a sample with local audio
local_samples = df[df["audio_path"].notna()]
if len(local_samples) > 0:
    sample_row = local_samples.iloc[0]
    audio_path = sample_row["audio_path"]
    data, sr = sf.read(audio_path)
    print(f"Call ID: {sample_row['call_id']}, Sample Rate: {sr} Hz, Duration: {len(data)/sr:.2f}s")
    print(f"Transcript: {sample_row['transcript']}")
    ipd.Audio(data, rate=sr)
```
