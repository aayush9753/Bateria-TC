# Model Architecture Design: TTS-Guided Target Speaker Extraction

**Date:** 2026-04-05  
**Version:** v1.0 (pre-ablation; to be revised after ablation phase)  

---

## 1. OVERVIEW

**Model name:** TTSE-Net (TTS-guided Target Speaker Extraction Network)

**Core idea:** Given a mixed waveform `x(t)` and a TTS-synthesized reference `ŝ(t)` (same words as target, approximate voice), extract the target speaker's clean speech `ŝ_extracted(t)`.

**Design philosophy:**
- Maximize signal quality on English first; extend to multilingual via representation fine-tuning
- Handle music and noise via band-aware backbone (BSRNN-inspired)
- Dual conditioning: speaker embedding (identity) + T-F template (spectral shape)
- Optional iterative refinement for quality ceiling (not required for streaming)
- Two deployment paths: high-quality (offline) and streaming (online) variants

---

## 2. INPUT / OUTPUT SPECIFICATION

### Input
| Parameter | Spec |
|---|---|
| Sample rate | 16 kHz (8 kHz for streaming telephony variant) |
| Format | Mono WAV, float32 |
| Mixture length | Arbitrary (chunked internally to 4-sec frames) |
| TTS reference length | 2–15 seconds |
| Feature type | Raw waveform (encoder handles featurization) |

### Output
| Parameter | Spec |
|---|---|
| Sample rate | 16 kHz |
| Format | Mono WAV, float32 |
| Latency (streaming mode) | < 200ms algorithmic delay |
| Latency (offline mode) | None (full file) |

---

## 3. HIGH-LEVEL ARCHITECTURE

```
                    ┌──────────────────────────────────────────┐
                    │              TTSE-Net                    │
                    │                                          │
  x(t) ────────────┤→ [Encoder] ──→ [Band Split] ──────────→  │
  (mixture)         │                     │                    │
                    │              [Sep Backbone]              │
  ŝ(t) ────────────┤→ [Speaker Encoder] ─┤ FiLM               │──→ ŝ_ext(t)
  (TTS reference)   │                     │                    │  (extracted)
                    │  [T-F Template] ────┤ Cross-Attn         │
                    │  (from ŝ(t) STFT)   │                    │
                    │                     ↓                    │
                    │              [Band Merge]                │
                    │                     │                    │
                    │              [Decoder]                   │
                    └──────────────────────────────────────────┘

[Optional Iterative Loop: extracted output → re-encode → better conditioning → repeat N=2 times]
```

---

## 4. COMPONENT-BY-COMPONENT DESIGN

### 4.1 Waveform Encoder (Multi-Scale)

**Adopted from:** SpEx+ multi-scale encoder concept  
**Purpose:** Convert waveform to learnable feature representations at 3 temporal resolutions

```
x(t) [16kHz]
  │
  ├──→ Conv1D(kernel=2.5ms=40, stride=20) → 256-dim features [short, 50Hz]
  ├──→ Conv1D(kernel=10ms=160, stride=20) → 256-dim features [medium, 50Hz]  
  └──→ Conv1D(kernel=20ms=320, stride=20) → 256-dim features [long, 50Hz]
  │
  └──→ Concatenate → Linear(768→512) → X ∈ R^{T×512}
```

**Why multi-scale:** Short windows capture fine phonemic detail; long windows capture prosodic structure. Critical for distinguishing similar speakers.

### 4.2 Band-Split Module (Music/Noise Robustness)

**Adopted from:** BSRNN band-split design  
**Purpose:** Split T-F representation into frequency bands for joint speech+music handling

```
X ∈ R^{T×512}
  │
  ↓ Compute STFT (window=25ms, hop=10ms) → complex spectrogram C ∈ C^{T×F}
  │
  Split F into B=12 bands (log-spaced):
    Band 0:   0–300 Hz     (vocal fundamental)
    Band 1:   300–600 Hz   (formants F1)
    Band 2:   600–1200 Hz  (formants F2)
    Band 3:   1200–2400 Hz (sibilants, high formants)
    Band 4–7: 2400–8000 Hz (harmonics, music)
    Band 8–11: > 8000 Hz   (presence, air)
  │
  Per-band: Linear projection → band_features ∈ R^{T×B×D_band} where D_band=64
```

### 4.3 Speaker Encoder (Multilingual, Synthesis-Robust)

**Base model:** ECAPA-TDNN (or WavLM-large-derived speaker encoder)  
**Purpose:** Extract speaker identity embedding robust to TTS synthesis artifacts

```
ŝ(t) [TTS reference, 2–15 sec]
  │
  ↓ FBank features (80-dim, 25ms/10ms)
  │
  ECAPA-TDNN (pre-trained on VoxCeleb2 + multilingual fine-tuned)
  │
  → Global speaker embedding e_spk ∈ R^{192}
  │
  → Additional fine-tuning on (TTS-synthesized, real) speaker pairs
    using contrastive loss: sim(TTS_emb, real_emb) maximized,
    sim(TTS_emb, different_speaker_emb) minimized
```

**Multilingual handling:** Fine-tune ECAPA-TDNN on IndicSUPERB + VoxCeleb2 + Kathbath speaker ID data. Shared encoder — no language-specific branches.

**Synthesis robustness:** Add synthesis-paired data during speaker encoder training:
- Generate TTS speech for 10,000 speakers from VoxCeleb2
- Train with triplet loss: (TTS_i, real_i, real_j) where i≠j

### 4.4 T-F Template Extractor

**Purpose:** Extract spectral shape patterns from ŝ(t) to serve as soft mask priors

```
ŝ(t) [TTS reference]
  │
  ↓ STFT (same params as mixture STFT)
  │
  |STFT(ŝ)| → magnitude spectrogram M_ref ∈ R^{T_ref×F}
  │
  ↓ 2D Conv stack (3 layers, kernel 3×3, stride 2×2 in time)
  │   [reduces time dimension; frequency preserved]
  │
  → Template embedding T_ref ∈ R^{T'×F×D_t} where D_t=32
  │
  [Note: T' << T_ref due to striding; used for cross-attention over mixture features]
```

**Why template + embedding:** The speaker embedding is timing-invariant (captures who is speaking). The T-F template captures the characteristic spectral patterns (formants, harmonics) that appear when this voice speaks — additional information beyond identity.

### 4.5 Separation Backbone (Band-Split Conformer-LSTM)

**Architecture:** Hybrid — Conformer for inter-band modeling, LSTM for temporal modeling

```
For each band b:
  Input: X_b ∈ R^{T×D_band}
  
  Block (×6 layers):
    ┌─────────────────────────────────────────────────┐
    │ 1. FiLM conditioning from e_spk:               │
    │    γ, β = Linear(e_spk) → 2×D_band             │
    │    X_b = γ * X_b + β                            │
    │                                                  │
    │ 2. Cross-attention with T-F template:           │
    │    Q = X_b, K = V = T_ref[:,b,:]                │
    │    X_b = X_b + Attention(Q, K, V)               │
    │    (soft: gated by reliability score of ŝ(t))  │
    │                                                  │
    │ 3. Temporal LSTM (intra-band):                  │
    │    X_b = LayerNorm(BiLSTM(X_b) + X_b)          │
    └─────────────────────────────────────────────────┘

Cross-band Conformer (×2 layers):
  Input: all bands concatenated → X_all ∈ R^{T×B×D_band}
  Self-attention across band dimension (inter-band modeling)
  Captures harmonic relationships across bands
```

**FiLM conditioning:** Feature-wise Linear Modulation — multiply features by γ and add β derived from speaker embedding. Applied at every layer so speaker conditioning is deep (not just input-level).

**Template reliability gating:**
```
g = sigmoid(Linear([e_spk, mean_pool(T_ref)]))  # scalar gate
cross_attn_output = g * Attention(Q, K, V)       # 0 when unreliable
```
When music or noise in ŝ(t) is high, gate suppresses template contribution.

### 4.6 Band Merge & Decoder

```
Band outputs: {X_0, X_1, ..., X_{B-1}} each ∈ R^{T×D_band}
  │
  Concatenate across bands → X_merged ∈ R^{T×(B×D_band)}
  │
  Linear projection → D_out = 512
  │
  Complex mask prediction: 
    M_real, M_imag = Linear(X_merged) → ∈ R^{T×F} each
    Complex mask M = M_real + j*M_imag
    
  Apply to mixture STFT:
    STFT_clean = M * STFT_mixture
    
  Inverse STFT (overlap-add) → waveform output
```

**Why complex masking:** Preserves phase information; critical for music-contaminated audio where magnitude-only masking produces artifacts.

---

## 5. ITERATIVE REFINEMENT LOOP (OFFLINE MODE)

```
Iteration 0: Run forward pass with ŝ(t) conditioning → ŝ_0(t)
Iteration 1: 
  - Re-extract speaker embedding from ŝ_0(t): e_spk_1 = SpeakerEncoder(ŝ_0(t))
  - Blend: e_spk_blend = 0.7 * e_spk_1 + 0.3 * e_spk_0
  - Re-run separation backbone with e_spk_blend → ŝ_1(t)
Iteration 2: same → ŝ_2(t)

Stopping criterion:
  - Max 3 iterations (empirical; beyond this, quality plateaus)
  - OR: cosine_sim(e_spk_{n}, e_spk_{n-1}) > 0.98 (converged)
```

**Why this helps:** The initial TTS enrollment has wrong timing/prosody. After one separation pass, the extracted audio has correct timing and is closer to the real speaker. Re-embedding improves speaker conditioning quality.

**Expected gain:** ~1–2 dB SI-SDR (estimated; to be validated in ablation).

---

## 6. FULL FORWARD PASS (ASCII DIAGRAM)

```
INPUT
  x(t) ─────────────────────────────────────────────────┐
  ŝ(t) ──┬───────────────────────────────────────────── │
         │                                              │
         ▼                                              ▼
   ┌─────────────┐                            ┌───────────────────┐
   │ Speaker Enc │                            │ Multi-Scale Enc   │
   │ (ECAPA-TDNN)│                            │ (Conv1D ×3 scales)│
   └──────┬──────┘                            └────────┬──────────┘
          │ e_spk ∈ R^192                              │ X ∈ R^{T×512}
          │                                            │
   ┌──────┴──────┐                                     │
   │  T-F        │ T_ref ∈ R^{T'×F×32}                │
   │  Template   │                                     │
   └──────┬──────┘                                     │
          │                                            ▼
          │                              ┌─────────────────────────┐
          │                              │   STFT → Band Split     │
          │                              │   B=12 bands            │
          │                              └──────────┬──────────────┘
          │                                         │
          │    ┌────────────────────────────────────┘
          │    │
          ▼    ▼
   ┌──────────────────────────────────────────────┐
   │         Band-Split Conformer-LSTM            │
   │  ┌──────────────────────────────────────┐    │
   │  │  Per-band block (×6):                │    │
   │  │  1. FiLM(e_spk)                      │    │
   │  │  2. CrossAttn(T_ref, gate=g)         │    │
   │  │  3. Temporal BiLSTM                  │    │
   │  └──────────────────────────────────────┘    │
   │  Cross-band Conformer (×2)                   │
   └──────────────────┬───────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Band Merge    │
              │ Complex Mask  │
              └───────┬───────┘
                      │
                      ▼
              Inverse STFT + OLA
                      │
                      ▼
              ŝ_extracted(t) ──────── [OPTIONAL LOOP: re-embed, refine ×1-2]
```

---

## 7. STREAMING MODE MODIFICATIONS

For streaming (< 200ms latency):

| Modification | Impact |
|---|---|
| Replace BiLSTM → unidirectional LSTM | Quality ~1 dB lower |
| Chunked processing: 200ms chunks with 50ms overlap | Algorithmic latency = 200ms |
| Replace Cross-band Conformer → linear attention | Removes O(B²) attention cost |
| Disable iterative refinement | No quality loss |
| Speaker encoder: pre-computed e_spk (not re-computed per chunk) | Negligible quality loss if enrollment is static |

**Streaming architecture name:** TTSE-Net-S (Streaming)  
**Target RTF (Real-Time Factor):** < 0.3 on single CPU core (A53) or < 0.02 on GPU.

---

## 8. MODEL SIZE & COMPUTE

| Component | Parameters | FLOPS/sec |
|---|---|---|
| Multi-scale encoder | 2M | Low |
| Speaker encoder (ECAPA) | 6M | Moderate |
| T-F template extractor | 1M | Low |
| Band-split conformer-LSTM (6 layers) | 18M | High |
| Cross-band conformer (2 layers) | 6M | Moderate |
| Decoder | 1M | Low |
| **Total** | **~34M** | — |

**Comparison:**
- SpEx+: ~30M parameters
- SepFormer: ~26M parameters
- TF-GridNet: ~14M parameters (computationally heavier per param due to T-F ops)
- Our model: ~34M (acceptable; streaming variant ~20M without cross-band conformer)

---

## 9. DESIGN ASSUMPTIONS & DECISIONS

| Decision | Rationale | Alternative Considered |
|---|---|---|
| ECAPA-TDNN speaker encoder | Best balance of accuracy and speed; well-studied | WavLM-large (better but 300M params) |
| Complex T-F masking | Avoids phase reconstruction artifacts with music | Magnitude masking (simpler, but artifact-prone) |
| FiLM over cross-attention for speaker cond. | FiLM is O(D) per layer; cross-attention is O(T²) | Full cross-attention (too slow for streaming) |
| 12 frequency bands | Matches musical octave structure; covers speech formants | 8 or 16 bands (ablation needed) |
| 3 temporal scales in encoder | SpEx+ finding: capturing 2.5/10/20ms scales critical | Single scale (simpler but weaker) |
| Template reliability gating | Prevents bad TTS from harming extraction | Hard mask (would fail on noisy TTS) |
| 2–3 refinement iterations | Diminishing returns beyond 3 (empirical from RVAE-EM) | 5+ iterations (excessive compute) |

---

*This design will be revised based on ablation results from `ablation_plan.md` before implementation begins.*
