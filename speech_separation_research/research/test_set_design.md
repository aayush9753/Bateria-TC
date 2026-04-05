# Evaluation Benchmark Design: TTS-Guided TSE Test Set

**Date:** 2026-04-05  
**Goal:** A reproducible, public-data-only benchmark covering all deployment conditions  

---

## 1. DESIGN PRINCIPLES

1. **Reproducibility:** All sources are publicly available (CC-BY, CC-BY-SA, or equivalent).
2. **Balanced difficulty:** Three tiers (Easy / Medium / Hard) per condition.
3. **Production representativeness:** Conditions mirror real-world broadcast and conversational audio.
4. **Multilingual parity:** Equal coverage for English, Hindi, code-switched.
5. **Fixed random seed for all mixtures** (`seed=42`); full mixing scripts committed to repo.

---

## 2. SOURCE DATA

### Speech Sources

| Source | Language | Speakers | Hours | License | URL |
|---|---|---|---|---|---|
| LibriSpeech test-clean | English | 40 | 5.4h | CC-BY 4.0 | openslr.org/12 |
| LibriSpeech test-other | English | 33 | 5.1h | CC-BY 4.0 | openslr.org/12 |
| VoxCeleb1 test | English | 40 | ~10h | CC-BY | robots.ox.ac.uk/~vgg/data/voxceleb |
| VCTK | English (multi-accent) | 110 | 44h | CC-BY 4.0 | datashare.ed.ac.uk |
| Kathbath test | 12 Indian langs (Hindi subset) | 100+ | ~100h | CC-BY 4.0 | indicvoices.ai4bharat.org |
| Shrutilipi | Hindi, Gujarati, others | 1000+ | 6400h total | CC-BY 4.0 | indicnlp.ai4bharat.org |
| IndicTTS | 13 Indian langs (Hindi) | 12 (TTS) | ~10h | Open research | iitm.ac.in/donlab/tts |
| MUCS 2021 test | Hindi+English codeswitched | 200+ | ~10h | Research | mucs2021.ai4bharat.org |
| MLS Hindi | Hindi | 100+ | 50h | CC-BY 4.0 | openslr.org/94 |

### Noise / Music Sources

| Source | Type | Duration | License |
|---|---|---|---|
| MUSAN | Music, noise, speech | 900h | CC-BY 4.0 |
| MusDB18-HQ | Isolated music stems | 10h | CC-BY-NC-SA 4.0 |
| FSD50k | Sound events + noise | 100h | CC-BY 4.0 |
| DEMAND | Real-world ambient noise | 3h | CC-BY-SA |
| RIR_Noises (BUT reverb DB) | Room impulse responses | — | CC-BY 4.0 |

### Interfering Speaker Sources
- LibriSpeech train-clean-360 (distinct from test speakers)
- VoxCeleb2 dev (distinct from VoxCeleb1 test)
- Kathbath train (distinct from test)

---

## 3. TEST CONDITIONS MATRIX

### 3.1 English Conditions

| ID | Condition | Target Source | Interference | SNR Range | Samples | Difficulty |
|---|---|---|---|---|---|---|
| EN-C1 | Clean 2-speaker | LibriSpeech test-clean | 1 LibriSpeech spk | 0–5 dB | 500 | Easy |
| EN-C2 | Clean 3-speaker | LibriSpeech test-clean | 2 LibriSpeech spks | -2–5 dB | 300 | Medium |
| EN-N1 | Noisy 2-speaker | LibriSpeech test-other | 1 spk + MUSAN noise | -5–5 dB | 500 | Medium |
| EN-N2 | Noisy 3-speaker | LibriSpeech test-other | 2 spks + MUSAN noise | -10–0 dB | 300 | Hard |
| EN-M1 | Music background | VCTK | MusDB18 instrument mix | 0–10 dB | 500 | Medium |
| EN-M2 | Music + speech | VCTK | MusDB18 + 1 other spk | -5–5 dB | 300 | Hard |
| EN-R1 | Reverberant | LibriSpeech test-clean | 1 spk + RIR | 0–10 dB | 300 | Medium |
| EN-A1 | All combined | VCTK multi-accent | 2 spks + music + noise + RIR | -10–5 dB | 200 | Hard |

**Total English:** ~2,900 samples

### 3.2 Hindi Conditions

| ID | Condition | Target Source | Interference | SNR Range | Samples | Difficulty |
|---|---|---|---|---|---|---|
| HI-C1 | Clean 2-speaker | Kathbath test | 1 Kathbath spk | 0–5 dB | 400 | Easy |
| HI-C2 | Clean 3-speaker | Kathbath test | 2 Kathbath spks | -2–5 dB | 200 | Medium |
| HI-N1 | Noisy 2-speaker | Shrutilipi test | 1 spk + MUSAN noise | -5–5 dB | 400 | Medium |
| HI-N2 | Noisy 3-speaker | Shrutilipi test | 2 spks + MUSAN noise | -10–0 dB | 200 | Hard |
| HI-M1 | Music background | MLS Hindi | MusDB18 mix | 0–10 dB | 400 | Medium |
| HI-M2 | Music + speech | MLS Hindi | MusDB18 + 1 spk | -5–5 dB | 200 | Hard |
| HI-A1 | All combined | Kathbath/Shrutilipi | 2 spks + music + noise | -10–5 dB | 150 | Hard |

**Total Hindi:** ~1,950 samples

### 3.3 Code-Switched Conditions (Hindi-English)

| ID | Condition | Target Source | Interference | SNR Range | Samples | Difficulty |
|---|---|---|---|---|---|---|
| CS-C1 | Clean 2-speaker | MUCS 2021 test | 1 MUCS spk | 0–5 dB | 300 | Easy |
| CS-N1 | Noisy | MUCS 2021 test | 1 spk + MUSAN | -5–5 dB | 300 | Medium |
| CS-M1 | Music + speech | MUCS 2021 test | MusDB18 + 1 spk | -5–5 dB | 200 | Medium |
| CS-A1 | All combined | MUCS 2021 test | 2 spks + music + noise | -10–5 dB | 150 | Hard |

**Total Code-Switched:** ~950 samples

### **TOTAL TEST SET: ~5,800 samples**

---

## 4. POPULATION DIVERSITY REQUIREMENTS

### Demographic Coverage (per language)
- **Gender:** ≥40% female speakers, ≥40% male speakers, ≥5% child (LibriSpeech children subset, Kathbath children)
- **Age groups:** Child (5–12), Young adult (18–35), Adult (35–60), Senior (60+) — estimated from metadata where available
- **Accents (English):** 
  - American English: VCTK (US accents), LibriSpeech
  - British English: VCTK (UK accents)
  - Indian English: VCTK (Indian English subset), VoxCeleb (Indian diasporic)
  - Australian: VCTK

### Accent Tagging
All samples tagged with speaker accent label where available. Per-accent breakdown reported in evaluation.

---

## 5. MIXTURE GENERATION PROCEDURE

### 5.1 Software Stack
```python
# Core tools
import pyroomacoustics as pra    # RIR simulation
import scaper                    # Soundscape synthesis  
import soundfile as sf
import numpy as np
import random

# All reproducible with: random.seed(42), np.random.seed(42)
```

### 5.2 Mixing Protocol

```
For each test sample:

1. Sample target utterance (4–10 sec) from target source
2. Sample interfering content:
   - 2-spk: 1 utterance from interferer pool (non-overlapping speaker IDs)
   - 3-spk: 2 utterances from interferer pool
   - Music: random 10-sec segment from MusDB18 non-vocal stems
   - Noise: random 10-sec segment from MUSAN noise category

3. Apply room impulse response (RIR) if reverberant condition:
   - Sample RIR from BUT Reverb DB
   - Convolve target and interferers separately
   
4. Normalize each source to 0 dBFS RMS independently

5. Mix at target SNR:
   - SNR defined as target_RMS / interference_RMS
   - Interference = sum of all non-target sources
   - Draw SNR uniformly from [SNR_min, SNR_max] for condition
   
6. Clip to [-1, 1]; save as float32 WAV at 16kHz

7. Save metadata JSON:
   {
     "mixture_id": "EN-C1-00001",
     "target_speaker_id": "...",
     "target_file": "...",
     "interferer_ids": [...],
     "snr_db": 2.3,
     "reverb": false,
     "rir_file": null,
     "difficulty": "easy"
   }
```

### 5.3 TTS Enrollment Generation
For each test sample, generate the TTS enrollment `ŝ(t)`:
```
1. Get transcript of target utterance (ground-truth text)
2. Voice clone target speaker using a 5-sec enrollment clip 
   (distinct from the test utterance)
3. Synthesize ŝ(t) = TTS(transcript, voice_clone_embedding)
4. Repeat for 3 TTS systems: XTTS-v2, YourTTS, OpenVoice
5. Save all three ŝ(t) variants with the test sample
```

---

## 6. EVALUATION METRICS

### Signal Quality Metrics
| Metric | Tool | Measures | Threshold (good) |
|---|---|---|---|
| SI-SDR | `fast_bss_eval` | Signal-to-distortion ratio (scale-invariant) | > 15 dB |
| SDRi | `fast_bss_eval` | Improvement in SDR over mixture | > 10 dB |
| PESQ | `pesq` package | Perceptual quality (narrowband or wideband) | > 3.0 |
| ESTOI | `pystoi` | Short-time intelligibility | > 0.85 |

### Intelligibility Metric
| Metric | Tool | Measures |
|---|---|---|
| WER | Whisper large-v3 | Transcription accuracy of extracted speech vs. ground truth |
| CER | Whisper large-v3 (Hindi/CS) | Character error rate for Hindi/code-switched |

*Note: Use Whisper large-v3-turbo for efficiency; run on GPU batch size 32.*

### Perceptual Quality Metric
| Metric | Tool | Measures |
|---|---|---|
| DNSMOS P.835 | ONNX model (Microsoft) | Perceptual quality: SIG, BAK, OVRL sub-scores |
| UTMOS | utmos package | MOS prediction for naturalness |

### Speaker Similarity Metrics
| Metric | Tool | Measures |
|---|---|---|
| Cosine similarity (ECAPA) | SpeechBrain ECAPA-TDNN | Speaker identity preservation |
| Cosine similarity (WeSpeaker) | WeSpeaker ResNet34 | Alternative speaker similarity |
| Equal Error Rate proxy | Both models | % samples where extracted speaker ≠ target |

### Per-Condition Aggregation
- Report mean ± std for all metrics per condition ID
- Report per-difficulty tier summary
- Report per-language summary
- Report per-music/noise/clean summary
- Statistical significance test (paired t-test, α=0.05) when comparing models

---

## 7. TEST SET PACKAGING

### Directory Structure
```
test_set/
├── README.md
├── generate_mixtures.py          # Fully reproducible mixture generation
├── evaluate_model.py             # Unified evaluation script
├── metadata/
│   ├── en_conditions.csv
│   ├── hi_conditions.csv
│   └── cs_conditions.csv
├── mixtures/
│   ├── EN-C1/  (mixture WAVs + metadata JSONs)
│   ├── EN-C2/
│   ├── ...
│   └── CS-A1/
├── references/                   # Ground-truth target speech
├── tts_enrollment/
│   ├── xtts_v2/
│   ├── yourtts/
│   └── openvoice/
└── results/                      # Model output storage
```

### Metadata Schema (per sample)
```json
{
  "id": "EN-C1-00042",
  "language": "en",
  "condition": "EN-C1",
  "difficulty": "easy",
  "target_speaker_id": "1284",
  "target_gender": "female",
  "target_accent": "american",
  "target_duration_sec": 5.3,
  "n_interferers": 1,
  "has_music": false,
  "has_noise": false,
  "has_reverb": false,
  "snr_db": 2.1,
  "tts_enrollment": {
    "xtts_v2": "tts_enrollment/xtts_v2/EN-C1-00042.wav",
    "yourtts": "tts_enrollment/yourtts/EN-C1-00042.wav",
    "openvoice": "tts_enrollment/openvoice/EN-C1-00042.wav"
  },
  "mixture_file": "mixtures/EN-C1/EN-C1-00042_mix.wav",
  "reference_file": "references/EN-C1/EN-C1-00042_ref.wav"
}
```

---

## 8. COMPUTE ESTIMATES FOR EVALUATION

| Step | Tool | Time/Sample | Total (5800 samples) | GPU? |
|---|---|---|---|---|
| Mixture generation | pyroomacoustics | ~0.1s | ~10 min | No |
| TTS enrollment synthesis | XTTS-v2 | ~2s | ~3.2h × 3 systems = 10h | Yes |
| Model inference | TBD | ~0.5s | ~50 min per model | Yes |
| WER computation (Whisper) | GPU batch 32 | ~0.3s | ~30 min | Yes |
| DNSMOS | ONNX CPU | ~0.1s | ~10 min | No |
| ECAPA similarity | GPU batch 128 | ~0.05s | ~5 min | Yes |
| Full eval pipeline | — | — | ~2–3h per model | Yes (A100) |

---

*This test set, once built, constitutes a publishable benchmark contribution independent of the model.*
