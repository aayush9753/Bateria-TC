# Test Set Design v2: TTS-Guided TSE Evaluation Suite
**Date:** 2026-04-06  
**Based on:** Gap analysis v2, approach comparison v2, PDF-extracted benchmark specifications

---

## DESIGN PHILOSOPHY

The evaluation suite must answer five distinct questions:
1. Does CMHA conditioning on TTS enrollment actually work (vs. oracle real enrollment)?
2. Does the system handle music backgrounds (the real-world case)?
3. Does it generalize to Hindi and code-switched speech?
4. Does iterative refinement help (TTS→extracted pass 1→pass 2)?
5. How does performance degrade with acoustic challenge (SNR, RT60, speakers)?

Each test track isolates one variable. All tracks use the same base extractor model.

---

## TRACK 1: CORE EXTRACTION (Standard Conditions)

### Purpose
Establish baseline against published SOTA on standard benchmarks, enabling direct comparison with SpEx+, USEF-TSE, SepReformer, LauraTSE.

### Track 1A: WSJ0-2mix (Reference Benchmark)
- 3,000 test utterances (standard test split)
- 2 speakers, min/max SIR per WSJ0-2mix protocol
- 8 kHz resampled to 24 kHz for our model
- **Reference condition:** Oracle real enrollment (3-second held-out utterance from same speaker)
- **Enrollment variants:** 
  - `oracle`: Real held-out utterance (upper bound)
  - `tts_matched`: TTS synthesis of target speaker's words using CosyVoice voice clone
  - `tts_zero`: TTS synthesis using generic voice (no voice cloning)
  - `tts_f5`: Same words, F5-TTS synthesizer
  - `tts_cosyvoice2`: Same words, CosyVoice 2 synthesizer
- **Metrics:** SI-SDRi, SDRi, PESQ, STOI, WavLM-Sim

### Track 1B: LibriMix-2mix-clean
- Standard LibriMix 2-speaker clean test split (3,000 utterances)
- 16 kHz → 24 kHz
- Same enrollment variants as Track 1A
- **Additional:** oracle vs. tts_matched SI-SDRi gap (primary TTS enrollment delta metric)

### Track 1C: LibriMix-2mix-noisy (WHAMR!)
- 3,000 utterances with environmental noise + reverb
- RT60 0.1–0.6s, SNR -6 to +6 dB
- Tests TTS enrollment robustness to acoustic degradation

### Track 1D: LibriheavyMix (Challenging)
- 1,000 utterances (random sample from test split)
- Up to 8 simultaneous speakers, heavy reverb (RT60 up to 1.2s)
- Long-form (10–30s utterances)
- This is the hardest published speech-only benchmark

---

## TRACK 2: ENROLLMENT QUALITY ABLATION

### Purpose
Isolate the effect of enrollment quality on extraction — the core scientific question. This track has no parallel in prior work.

### 2A: TTS Quality Gradient
Five enrollment conditions covering the quality spectrum:

| Condition | Description | Expected SI-SDR |
|-----------|-------------|-----------------|
| `oracle_real` | Real recording of target, same content | Upper bound |
| `oracle_different` | Real recording, different words | Should match oracle_real |
| `tts_voice_clone` | TTS with correct voice clone (CosyVoice 2) | Primary evaluation |
| `tts_correct_words_wrong_voice` | TTS correct text, wrong speaker voice | Tests CMHA robustness |
| `tts_wrong_words_correct_voice` | TTS wrong text, correct voice | Isolates lexical vs. voice |
| `tts_generic_voice` | TTS with no speaker adaptation | Lower bound |

**N:** 500 utterances per condition (3,000 total)  
**Source:** WSJ0-2mix speakers, TTS generated offline

### 2B: Enrollment Duration Effect
- 0.5s, 1s, 2s, 3s, 5s, 10s enrollment duration
- N: 200 utterances × 6 durations = 1,200
- Motivation: Short TTS output may be less reliable for CMHA conditioning

### 2C: TTS Timing Mismatch Analysis
- Enrollment generated at different speaking rates: 0.6×, 0.8×, 1.0×, 1.2×, 1.5× normal speed
- Tests CMHA's timing invariance claim
- N: 200 × 5 = 1,000

### 2D: Iterative Refinement Protocol
- Run extraction pass 0 (TTS enrollment) → ŝ₁
- Run extraction pass 1 (ŝ₁ as enrollment) → ŝ₂  
- Run extraction pass 2 (ŝ₂ as enrollment) → ŝ₃
- Compare SI-SDR and WavLM-Sim at each pass
- N: 500 utterances, 3 passes each
- **Key metric:** Δ(pass 1 → oracle_real) / Δ(pass 0 → oracle_real) — refinement efficiency

---

## TRACK 3: MUSIC BACKGROUNDS (MusicMix-TSE Benchmark)

### Purpose
First published evaluation of TSE with music backgrounds. Creates a new benchmark that can be adopted by future work.

### Data Construction
**Speech source:** LibriSpeech test-clean (468 utterances, 5.4 hours), normalized to -26 LUFS  
**Music source:** MedleyDB test split (100 tracks), MusDB18 test (50 tracks), FMA-small test (100 tracks)  
**Mixing:** Randomly select 30s segments from each source, mix at specified SIR

### 3A: Genre × SNR Grid

| Genre | SIR = -5 dB | SIR = 0 dB | SIR = +5 dB | SIR = +10 dB |
|-------|------------|-----------|------------|-------------|
| Pop/Rock (MusDB18) | 75 | 75 | 75 | 75 |
| Electronic/EDM (FMA) | 75 | 75 | 75 | 75 |
| Classical (MedleyDB) | 75 | 75 | 75 | 75 |
| Ambient/Drone (FMA) | 75 | 75 | 75 | 75 |

**Total:** 1,200 utterances  

### 3B: Speech+Music+Speaker Overlap
- Two speakers + music background (3-source mixture)
- Same SIR grid but with second speaker at +3 dB relative to music
- N: 600 utterances

### 3C: Music with Vocals
- Speech+music-with-vocals (most confusing case — two vocal sources)
- Music source: "full mix" tracks from MusDB18 test (includes singing)
- N: 300 utterances
- Expected: large degradation if model can't distinguish speech from singing

### Metrics for Track 3
- **SI-SDRi** (primary, compared to speech-only baseline)
- **Music residual** (energy of extracted output in music frequency bands: 200–3500 Hz, compared to ground truth)
- **Spectral artifacts** (perceptual evaluation: DNSMOS-P.835 on extracted speech)
- **Listening test** (N=50 random samples, 5-point MOS for music-bleed artifacts)

---

## TRACK 4: MULTILINGUAL

### Purpose
Evaluate performance on Hindi monolingual, English monolingual (matched), and code-switched conditions. First such evaluation in TSE literature.

### Data Sources
- **Hindi:** Kathbath test split (clean + noisy), MUCS 2021 far-field track
- **English:** Matched subset of LibriSpeech test-clean
- **Code-switched:** 
  - LIMMITS 2023 challenge data (EN-HI)
  - Manually constructed: 50 speakers, 3 utterances each with code-switching
  - Mixing: synthesized by concatenating EN and HI speech from same speaker

### 4A: Hindi Monolingual
- 2-speaker Hindi mixtures from Kathbath test
- Construction: pair utterances from different speakers in Kathbath test set
- Enrollment: real (oracle), TTS (Indic-TTS / CosyVoice-Hindi)
- N: 500 utterances
- SNR range: 0, 5, 10 dB

### 4B: Cross-Lingual (Reference in One Language, Mixture in Another)
| Condition | Reference Language | Mixture Language |
|-----------|--------------------|-----------------|
| EN-ref → EN-mix | English | English |
| HI-ref → HI-mix | Hindi | Hindi |
| EN-ref → HI-mix | English | Hindi |
| HI-ref → EN-mix | Hindi | English |
| TTS-EN-ref → HI-mix | TTS English | Hindi |
| TTS-HI-ref → EN-mix | TTS Hindi | English |

N: 200 utterances per condition (1,200 total)  
Key question: Does WavLM-based CMHA generalize across languages?

### 4C: Code-Switched Speech
- Speaker switches between Hindi and English mid-utterance
- Two types: (a) target is code-switched, reference is English-only; (b) both are code-switched
- N: 300 utterances
- Enrollment: TTS code-switched (hard), TTS English-only (easier), oracle

### Metrics for Track 4
- **SI-SDRi** (primary)
- **WavLM-Sim** (cross-lingual, using multilingual-WavLM)
- **dWER** (Whisper-large-v3 for English, IndicWhisper for Hindi)
- **DNSMOS-P.835** (perceptual quality)

---

## TRACK 5: ACOUSTIC CHALLENGE CONDITIONS

### Purpose
Stress-test the system under degraded conditions that occur in real deployment.

### 5A: Reverb Sweep
- RT60: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5 s
- RIR generation: pyroomacoustics, random room sizes 4×4×3 to 10×8×4 m
- N: 150 × 7 = 1,050

### 5B: SNR Sweep
- SIR (between target and interference): -10, -5, 0, 5, 10, 15 dB
- Noise type: white, babble, street, WHAM! noise
- N: 100 × 6 × 4 = 2,400

### 5C: Overlapping Duration Sweep
- Overlap ratio: 0%, 25%, 50%, 75%, 100%
- 0% = turn-taking (no simultaneous speech), 100% = full overlap
- N: 200 × 5 = 1,000

### 5D: Number of Speakers
- 2, 3, 4, 5, 6 simultaneous speakers
- All interferers at approximately equal loudness
- N: 100 × 5 = 500

### 5E: Channel Mismatch (Telephone / Mobile)
- Bandwidth limitation: 8kHz (telephone), 16kHz (wideband), 24kHz (fullband)
- Codec degradation: G.711 µ-law, MP3 64kbps, Opus 32kbps
- Background noise types: office, car, street
- N: 100 × 3 × 3 × 3 = 2,700

---

## TRACK 6: TTS SYSTEM COMPARISON

### Purpose
Test whether extraction quality depends on which TTS system generates the enrollment.

### 6A: TTS System Ablation

| TTS System | Voice Cloning | Languages | Notes |
|------------|--------------|-----------|-------|
| CosyVoice 2 | 3-shot | EN, HI, CS | Best multilingual |
| F5-TTS | 15s reference | EN | Fast inference |
| YourTTS | Seen/unseen | EN | Zero-shot |
| XTTS v2 | Few-shot | EN | Coqui open-source |
| Parler-TTS | Description | EN | No cloning |
| Oracle (real speech) | N/A | EN | Upper bound |

N: 500 utterances × 6 TTS systems = 3,000  
Key metric: SI-SDR gap between TTS-system and oracle enrollment

### 6B: Voice Clone Quality
- Clone quality measured by MOS (1–5 scale, human evaluation)
- Correlate clone MOS with extraction SI-SDR
- Hypothesis: CMHA is robust to voice clone quality if spectral structure is preserved

### 6C: Seen vs. Unseen Speaker
- Seen: speakers in TTS training set (voice clone quality higher)
- Unseen: speakers NOT in TTS training set (zero-shot cloning)
- N: 250 seen + 250 unseen per TTS system

---

## TRACK 7: MODEL COMPARISON TRACK

### Purpose
Position our model against published SOTA. All comparisons use same test conditions.

### Models to Compare

| Model | Type | Params | Reference |
|-------|------|--------|-----------|
| SpEx+ | Discriminative | 28M | TASLP 2020 |
| VoiceFilter | Discriminative | 8M | Interspeech 2018 |
| USEF-TSE (TFGridNet) | Disc. (CMHA) | 120M | ICASSP 2024 |
| SepReformer | Discriminative | 98M | NeurIPS 2024 |
| LauraTSE | Generative (AR-LM) | ~400M | ICASSP 2025 |
| AnyEnhance | Generative (MaskGIT) | ~300M | 2025 |
| TTSE-Net-Track1 | Disc. (CMHA+Mamba) | 750M | Ours |
| TTSE-Net-Track2 | Flow matching (DiT) | 2.5B | Ours |

### Comparison Conditions
1. **Standard:** LibriMix-2mix-clean, oracle enrollment
2. **TTS enrollment:** LibriMix-2mix-clean, TTS-matched enrollment
3. **Music:** MusicMix-TSE at 0 dB SIR
4. **Hindi:** Kathbath-2mix, oracle enrollment
5. **Hard:** LibriheavyMix, oracle enrollment

---

## TOTAL SAMPLE COUNTS

| Track | Utterances | Hours (est.) |
|-------|-----------|-------------|
| Track 1 (Core) | ~10,300 | ~14h |
| Track 2 (Enrollment Ablation) | ~5,700 | ~8h |
| Track 3 (Music) | 2,100 | ~4h |
| Track 4 (Multilingual) | 2,000 | ~4h |
| Track 5 (Acoustic Challenge) | 7,650 | ~11h |
| Track 6 (TTS System) | 3,000 | ~4h |
| Track 7 (Model Comparison) | 3,000 | ~4h |
| **TOTAL** | **~33,750** | **~49h** |

---

## EVALUATION METRICS REFERENCE

### Objective (automated)

| Metric | Range | Better | What it measures |
|--------|-------|--------|------------------|
| SI-SDRi | -∞ to ~25 dB | Higher | Improvement in scale-invariant SNR |
| SDRi | -∞ to ~25 dB | Higher | Classical SDR improvement |
| PESQ-WB | 1–4.5 | Higher | Narrowband speech quality |
| STOI | 0–1 | Higher | Intelligibility |
| DNSMOS-P.835 | 1–5 | Higher | Perceptual speech quality |
| WavLM-Sim | 0–1 | Higher | Speaker identity preservation |
| MOS-UTMOS | 1–5 | Higher | Overall naturalness |
| dWER | 0–∞ | Lower | ASR degradation |
| Music Residual | dB | Lower | Music bleed-through in extracted |

### Subjective (human listening)

| Test | Scale | N evaluators | N samples |
|------|-------|-------------|----------|
| MOS (overall) | 1–5 | 20 | 200 |
| MUSHRA (quality vs. ref) | 0–100 | 20 | 100 |
| Speaker similarity (ABX) | % correct | 30 | 150 |
| Music artifact (7-point) | 1–7 | 20 | 100 |

---

## REPRODUCIBILITY REQUIREMENTS

1. **All test mixtures generated with fixed random seeds (seed=42)** and stored as wav files (24kHz, 16-bit)
2. **All TTS enrollments pre-generated** offline before evaluation (no randomness at eval time)
3. **Evaluation scripts** published alongside model weights
4. **Manual listening test** conducted via blind ABX protocol on 3 annotators per sample

### Data Release Plan
- MusicMix-TSE: Release generation scripts + metadata (no copyrighted audio)
- Multilingual test set: Release Kathbath-derived mixtures under original CC-BY license
- All enrollment wavs: Release TTS-generated enrollments (no license issues)
- Model checkpoints: Apache 2.0

---

## PILOT EVALUATION (SANITY CHECK BEFORE FULL EVAL)

Before running full evaluation suite, run pilot on 50 utterances per track:
1. Verify SI-SDR > 0 for all oracle conditions (sanity: model extracts something)
2. Verify TTS enrollment SI-SDR < oracle (sanity: TTS is harder than real)
3. Verify iterative refinement improves by at least 0.5 dB on pass 1
4. Verify music-SI-SDR < speech-only-SI-SDR (sanity: music is harder)
5. Verify Hindi SI-SDR > -∞ (sanity: multilingual model doesn't crash)
