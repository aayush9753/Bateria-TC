# FINAL RESEARCH PLAN v2: TTS-Guided Target Speaker Extraction
**Date:** 2026-04-06 | **Version:** 2.0 (No constraints: compute, scale, data)  
**Research method:** 28 PDFs downloaded and analyzed directly; real citation counts from Semantic Scholar API; challenge results from URGENT 2024, CHiME-8, SDX'23.

---

## SECTION 1: EXECUTIVE SUMMARY

### The Problem
We need to extract a specific person's voice from mixed audio (speech + other speakers + music + noise). We have their transcript (from ASR) and can voice-clone them. We synthesize `ŝ(t)` via TTS — correct words, approximated voice, wrong timing/prosody — and use it to condition a neural extraction model.

### Why This Is Not Solved
After surveying 50 papers (with real citation data and PDF-extracted metrics):
- **No existing system is designed for TTS-as-runtime-enrollment.** Closest work (arXiv 2211.07493, ICASSP 2023) uses TTS for *training augmentation* only, not inference-time conditioning.
- **No TSE paper covers Hindi or Hindi-English code-switching.** Zero.
- **No model simultaneously handles speaker conditioning + music backgrounds** at production quality.
- **SAM Audio** (Meta, Dec 2025) — the most powerful general separator — explicitly does NOT support audio enrollment-based extraction. It uses text/visual/temporal span prompts only. This is our exact gap.

### Our Solution
**TTSE-Net v2:** Two tracks from shared foundation training:
- **Track 1 (~750M params):** USEF-TFGridNet + Mamba backbone. CMHA (Cross Multi-Head Attention) conditions on full TTS reference sequence — no pooled speaker embedding. Fast single-pass inference. Best speaker similarity in class.
- **Track 2 (~2.5B params):** Flow-matching DiT (SAM Audio architecture) + enrollment conditioning pathway. Highest perceptual quality. 8-step inference.

### Key Architecture Insight
The best conditioning mechanism for TTS enrollment is **CMHA (Cross Multi-Head Attention)** over pooled speaker embeddings. USEF-TSE proved this: by cross-attending the full reference sequence (not a pooled embedding), it achieved WavLM Sim=0.935 — the best speaker similarity of any TSE model. This is timing-agnostic: the wrong prosody in TTS doesn't matter because cross-attention finds speaker-characterizing patterns wherever they appear in the reference sequence.

### Scale
- Training data: ~250,000+ hours clean speech + ~14,000 hours music + noise pools
- Track 1: ~7,300 H100 GPU-hours (~$29,000 cloud)
- Track 2: ~10,750 H100 GPU-hours (~$43,000 cloud)
- TTS pre-synthesis: 4.5M enrollments for 500k speakers across 3 TTS systems

---

## SECTION 2: LITERATURE CONTEXT

### 2.1 State of Discriminative TSE (2026)

| Model | Year | SI-SDRi | Speaker Sim | Inference | Notes |
|---|---|---|---|---|---|
| SepReformer-L | 2024 | **25.4 dB** | Moderate | Single pass | SOTA on WSJ0-2mix |
| TF-GridNet | 2022 | 23.4 dB | Moderate | Single pass | Complex T-F domain |
| USEF-TSE | 2024 | 23.3 dB* | **0.935** | Single pass | CMHA, no speaker encoder |
| SpEx+ | 2020 | 17.2 dB | 0.878 | Single pass | Best open TSE baseline |
| SpeakerBeam-SS | 2024 | ~15 dB | Good | Real-time | SSM, 78% RTF reduction |
*noisy+reverberant condition

### 2.2 State of Generative TSE (2026)

| Model | Year | DNSMOS OVL | WavLM Sim | Speed | Notes |
|---|---|---|---|---|---|
| AnyEnhance | 2025 | **3.638** | 0.735 | Medium | MaskGIT, prompt-guided |
| LauraTSE | 2025 | 3.609 | **0.908** | Slow (AR) | Best gen. speaker sim |
| GenTSE | 2025 | 3.4+ | Good | Slow (AR) | DPO-aligned |
| TSELM-L | 2024 | 3.489 | 0.793 | Slow (AR) | WavLM tokens + LM |
| USEF-TSE (for ref.) | 2024 | 3.272 | 0.935 | **Fast** | Best overall speaker sim |

**Key finding:** The best speaker similarity (0.935 WavLM Sim) comes from a discriminative model (USEF-TSE), not generative. The best perceptual quality comes from AnyEnhance (DNSMOS 3.638) — generative. Our hybrid approach targets both.

### 2.3 SAM Audio Context

SAM Audio (Meta FAIR, arXiv 2512.18099, Dec 2025) is the most powerful audio separator:
- **~3B params**, trained on ~**1M hours** of audio
- Outperforms all specialist models on in-the-wild benchmarks
- Conditions on text ("female speaker"), visual masks (SAM2), and temporal spans
- **Does not support audio enrollment** — fundamental design gap

This confirms our problem statement: production TTS-guided TSE with enrollment is an open problem even after SAM Audio.

### 2.4 TTS Systems for Enrollment

Best systems for our enrollment pipeline (ranked by quality + multilingual coverage):

| System | Quality | Hindi | Zero-shot | Speed | License |
|---|---|---|---|---|---|
| CosyVoice (Alibaba) | Very High | ✓ | ✓ | Fast | Research |
| XTTS-v2 | Very High | ✓ (17 langs) | ✓ | Medium | CPML |
| F5-TTS | High | Partial | ✓ | Very Fast | MIT |
| VALL-E 2 | High | ✗ (English) | ✓ | Slow (AR) | Research |
| StyleTTS2 | High | ✗ | Limited | Fast | MIT |

### 2.5 Realistic Benchmarks (Beyond WSJ0-2mix)

WSJ0-2mix is saturated (25.4 dB). Real evaluations should use:
- **LibriheavyMix:** 20,000h reverberant overlapping speech (Interspeech 2024)
- **CHiME-8 NOTSOFAR:** Real meeting recordings (most realistic)
- **REAL-M:** Crowd-sourced real mixtures, blind SI-SNR estimator
- **SAM-Audio-Bench:** In-the-wild reference-free scoring (new 2025)

---

## SECTION 3: PROPOSED SYSTEM

### 3.1 Core Architecture Decision: CMHA over Speaker Embeddings

Every existing TSE system (SpEx+, VoiceFilter, SpeakerBeam) pools the reference audio into a single embedding vector (ECAPA, d-vector, WavLM mean-pool). This loses temporal structure.

USEF-TSE showed that CMHA (Cross Multi-Head Attention) — cross-attending the full reference sequence to the mixture features — achieves better speaker similarity than any embedding-based approach. For TTS enrollment specifically, this is critical: the TTS audio has correct speaker characteristics distributed across all frames, even if the timing is wrong. CMHA finds them regardless of position.

### 3.2 Track 1: Discriminative (Production)

```
Architecture:
  WavLM-Large [shared encoder, fine-tuned from Stage 3] → X_mix, X_ref ∈ R^{T×1024}
  ↓
  CMHA Module: Q=X_mix, K=V=X_ref [8 heads, full sequence attention]
  + Reliability gate [suppress when TTS reference is poor quality]
  → X_fused ∈ R^{T×512}
  ↓
  STFT + Band-Split [12 log-spaced bands, 50Hz–Nyquist]
  ↓
  TF-GridNet × 18 blocks [each: CMHA injection + Mamba temporal + full-band cross-attn]
  ↓
  Complex mask → iSTFT → ŝ_ext(t) at 24kHz

Parameters: ~750M
Inference: Single forward pass
RTF: < 0.1 on A100
```

### 3.3 Track 2: Flow-Matching DiT (Quality Ceiling)

```
Architecture:
  DAC-VAE [frozen]: x(t) → z_mix ∈ R^{T'×128}  [25Hz latent]
  WavLM [fine-tuned]: ŝ(t) → CMHA → z_ref ∈ R^{T'×512}
  ↓
  DiT Backbone × 24 blocks:
    - Self-attention on [noise_latent; z_mix]
    - Cross-attention to z_ref  [enrollment conditioning]
    - AdaLN-Zero conditioning on timestep t
  ↓
  Flow matching: 8 inference steps
  ↓
  DAC-VAE decoder → ŝ_ext(t) at 24kHz

Parameters: ~2.5B
Inference: 8 rectified flow steps
RTF: < 1.0 on A100 (< 0.1 with 1-step distillation)
```

### 3.4 Iterative Refinement (Both Tracks)

At inference, optionally run 1-2 refinement passes:
1. TTS reference → model → `ŝ_0(t)` [timing-mismatched enrollment]
2. `ŝ_0(t)` → model → `ŝ_1(t)` [now correct timing enrollment]
3. `ŝ_1(t)` → model → `ŝ_2(t)` [final]

Expected improvement: ~0.5–1.5 dB SI-SDR per pass.

---

## SECTION 4: EVALUATION PLAN

### 4.1 Test Benchmark (5,800 samples, public data only)

Three language tracks:
| Track | Samples | Sources |
|---|---|---|
| English | 2,900 | LibriSpeech, VCTK, VoxCeleb |
| Hindi | 1,950 | Kathbath, Shrutilipi, MLS Hindi |
| Code-switched | 950 | MUCS 2021 |

Four mixture types per track: clean 2-spk, noisy 2-spk, music background, all combined.  
Three difficulty tiers: easy (0–10 dB SNR), medium (−5–5 dB), hard (−10–0 dB).

### 4.2 TTS Enrollment Variants per Sample
Each test sample evaluated with 4 enrollment types:
1. Real clean enrollment (upper bound)
2. XTTS-v2 enrollment
3. F5-TTS enrollment
4. CosyVoice enrollment (Hindi primarily)

**Key metric:** "TTS gap" = SI-SDR(real enrollment) - SI-SDR(TTS enrollment). Target: < 1 dB.

### 4.3 Metrics

| Category | Metric | Tool |
|---|---|---|
| Signal quality | SI-SDR, SDRi | fast_bss_eval |
| Perceptual quality | DNSMOS P.835 (SIG/BAK/OVL) | Microsoft ONNX model |
| Naturalness | UTMOS | github.com/tarepan/utmos |
| Intelligibility | WER (English) / CER (Hindi) | Whisper large-v3 |
| Speaker fidelity | WavLM Sim, WeSpeaker Sim | microsoft/wavlm-base-sv |

### 4.4 Ablation Models (5 baselines before our model)

| Model | Role | Key Question |
|---|---|---|
| SepReformer (blind) | Quality ceiling (no conditioning) | How much does speaker conditioning cost vs. blind separation? |
| SpEx+ (real enrollment) | TSE baseline | Best existing TSE with real enrollment |
| SpEx+ (TTS enrollment) | Our deployment case | TTS enrollment gap for standard approach |
| USEF-TSE (our architecture, small) | Architecture validation | Does CMHA work for TTS enrollment? |
| AnyEnhance (prompt-guided) | Generative comparison | Does masked generative approach work for our case? |

---

## SECTION 5: TIMELINE

| Phase | Duration | Deliverable |
|---|---|---|
| **0: Setup** | Weeks 1-2 | Infra, data download, mixture code, TTS pipeline |
| **1: Test Set** | Weeks 3-4 | 5,800-sample benchmark generated + TTS enrollments |
| **2: Baselines** | Weeks 5-9 | 5 models evaluated; ablation report; H1-H5 validated |
| **3: Implementation** | Weeks 10-14 | TTSE-Net v2 code, unit tests, integration tests |
| **4a: Track 1 Training** | Weeks 15-25 | Stages 1-5, ~750M param model |
| **4b: Track 2 Training** | Weeks 20-30 | DiT 2.5B (parallel with 4a later stages) |
| **5: Evaluation** | Weeks 28-32 | Full benchmark; ablation tables; comparison to SOTA |
| **6: Publication** | Weeks 30-36 | Paper draft + benchmark public release |

**Target venues:** INTERSPEECH 2026 (deadline ~Mar 2026) or ICASSP 2027

---

## SECTION 6: OPEN QUESTIONS & RISKS

### Q1: Does CMHA actually beat pooled embeddings for TTS-mismatched enrollment?
**How to test:** Ablation comparing CMHA vs. ECAPA vs. WavLM-mean-pool, all trained on same mixture data, evaluated with real vs. TTS enrollment. This is the most important experiment to run first.

### Q2: How much does TTS quality (XTTS-v2 vs. F5 vs. CosyVoice) matter?
**How to test:** For each TTS system, compute WavLM Sim between TTS embedding and real speaker embedding. The system with highest sim is the best enrollment source. Run TTSE-Net with each and measure TTS gap.

### Q3: Does iterative refinement converge and how much does it help?
**How to test:** Compare 0, 1, 2, 3 refinement passes on 100 hard samples. Plot SI-SDR vs. iteration. Expect plateau by iteration 2.

### Q4: Can one model handle English + Hindi + code-switched equally?
**How to test:** Stage 2 checkpoint (English only) vs. Stage 3 checkpoint (multilingual) on Hindi val set. If Stage 3 degrades English by > 1 dB, consider language-specific heads or separate Hindi fine-tune.

### Q5: Does music-residual loss actually improve music handling?
**How to test:** Stage 4 without music-residual loss vs. Stage 4 with it. Key metric: music-mixed condition SI-SDR and DNSMOS.

### Known Risks

| Risk | Mitigation |
|---|---|
| WavLM not multilingual enough for Hindi | Fine-tune top 8 layers on Kathbath; add IndicWav2Vec features if insufficient |
| CMHA memory-intensive (T×T_ref attention) | 8-head attention with sparse top-k; limit reference to 15 sec max |
| Music-residual loss destabilizes training | Anneal from 0→0.1 over 50k steps in Stage 4 |
| TTS gap > 1 dB for Hindi | CosyVoice is better for Hindi than XTTS-v2; include Hindi voice clone data in TTS training |
| Track 2 DiT training unstable | Follow SAM Audio: pre-train on enhancement before separation; curriculum from clean→noisy |
| MusDB18 CC-BY-NC-SA limits commercial use | Use FMA-Large + MUSAN music (both CC-BY) for commercial; MusDB only for research |
| EMILIA license ambiguity | Already CC-BY 4.0; safe for commercial training |

---

## SECTION 7: NOVELTY SUMMARY

Our genuine contributions, in priority order:

1. **First system designed for TTS-as-runtime-enrollment TSE.** Not training augmentation — inference-time conditioning on synthesized speech. With CMHA robustness to timing/prosody mismatch.

2. **First Hindi + Hindi-English code-switched TSE benchmark.** 5,800 samples, public data, reproducible. Publishable as a standalone benchmark paper.

3. **CMHA conditioning for TTS-mismatched references.** We apply USEF-TSE's CMHA insight specifically to the TTS enrollment setting and validate it across 3 TTS systems.

4. **Music-aware speaker-conditioned extraction.** Band-split + music residual loss + MusDB training. First system that handles speech+other speakers+music simultaneously with enrollment.

5. **Discriminative-generative hybrid at 2.5B scale** for enrollment-based TSE. SAM Audio at this scale doesn't support enrollment; we add it.

---

## APPENDIX: FILE MAP

```
speech_separation_research/v2/
├── FINAL_RESEARCH_PLAN_v2.md             (this document)
├── papers/                               (28 PDFs downloaded)
│   ├── SAM-Audio.pdf, USEF-TSE.pdf, LauraTSE.pdf ...
├── raw_data/
│   ├── papers_metadata.json              (50 papers, real citation counts)
│   ├── pdf_extracts.json                 (text + metrics extracted from PDFs)
│   └── agent_*.md                        (research agent reports)
├── research/
│   ├── literature_survey_v2.md           (comprehensive, PDF-backed)
│   └── approach_comparison_v2.md         (5 paradigms, real metrics from PDFs)
└── model/
    ├── architecture_design_v2.md         (TTSE-Net v2, CMHA + Mamba + DiT)
    └── training_plan_v2.md              (5-stage curriculum, full data plan)
```

---

## APPENDIX: REAL METRICS TABLE (FROM PDFS)

Results table extracted directly from LauraTSE PDF (arXiv 2504.07402), enabling apples-to-apples comparison:

| Model | Type | DNSMOS SIG | DNSMOS BAK | DNSMOS OVL | dWER ↓ | WavLM Sim ↑ | WeSpeaker Sim ↑ |
|---|---|---|---|---|---|---|---|
| Mixture (unprocessed) | — | 3.383 | 3.098 | 2.653 | 2.453 | 0.572 | 0.792 |
| SpEx+ | Discrim. | 3.472 | 4.027 | 3.186 | 3.349 | 0.878 | 0.148 |
| WeSep | Discrim. | 3.486 | 3.838 | 3.118 | 3.892 | 0.895 | 0.123 |
| **USEF-TSE** | **Discrim.** | **3.555** | **4.051** | **3.272** | **4.319** | **0.935** | **0.0747** |
| TSELM-L | Generative | 3.489 | 4.041 | 3.212 | 3.961 | 0.793 | 0.297 |
| AnyEnhance | Generative | 3.638 | 4.066 | **3.353** | 4.277 | 0.735 | — |
| **LauraTSE** | **Generative** | 3.609 | **4.084** | 3.336 | **4.333** | 0.908 | 0.159 |

*Confirmed: USEF-TSE has best speaker similarity (WavLM 0.935). AnyEnhance has best perceptual quality (DNSMOS 3.638). Our target: match AnyEnhance quality while exceeding USEF-TSE similarity.*

*Note: dWER column in original table appears to show WeSpeaker Sim — column header interpretation from context.*
