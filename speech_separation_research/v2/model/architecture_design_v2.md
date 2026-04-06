# TTSE-Net v2: Architecture Design (Unconstrained)
**Date:** 2026-04-06  
**Constraint assumptions:** No model size limit, no compute limit, train from scratch, data can be purchased.  
**Based on:** Direct PDF analysis of USEF-TSE, LauraTSE, AnyEnhance, SAM Audio, SepReformer, Geneses, Disc-Gen-TSE.

---

## DESIGN PHILOSOPHY

Three key insights from the literature drive our design:

1. **CMHA beats pooled speaker embeddings for TTS enrollment.** USEF-TSE achieves WavLM Sim=0.935 (best in field) without any speaker encoder, by cross-attending the full reference sequence to the mixture. This is ideal for TTS-synthesized references because it is timing-agnostic.

2. **Discriminative-generative hybrids beat either paradigm alone.** Pure discriminative models (USEF-TFGridNet) win on speaker similarity and intelligibility. Pure generative models (AnyEnhance, LauraTSE) win on perceptual quality and DNSMOS. The Disc-Gen-TSE framework (arXiv 2601.06006) combines both.

3. **Scale solves multilingual and music.** SAM Audio (~1B params, ~1M hrs) handles arbitrary separation without task-specific design. We need the same scale but with enrollment conditioning.

---

## SYSTEM OVERVIEW

Two deployment variants from shared training:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TTSE-Net v2 System                           │
│                                                                 │
│  Input: x(t) = mixture    ŝ(t) = TTS reference (same words,    │
│                            voice-cloned, wrong timing/prosody) │
│                                                                 │
│  ┌─────────────────┐    ┌────────────────────────────────────┐  │
│  │  Track 1:       │    │  Track 2:                          │  │
│  │  Discriminative │    │  Flow-Matching DiT                 │  │
│  │  (Production)   │    │  (Quality Ceiling)                 │  │
│  │  ~1B params     │    │  ~3B params                        │  │
│  │  Single pass    │    │  8 flow steps                      │  │
│  │  RTF < 0.1      │    │  RTF < 1.0                         │  │
│  └─────────────────┘    └────────────────────────────────────┘  │
│         Both share: WavLM encoder + CMHA reference conditioning  │
└─────────────────────────────────────────────────────────────────┘
```

---

## TRACK 1: DISCRIMINATIVE USEF-TFGRIDNET AT SCALE

### 1.1 Input Specification
- Sample rate: 24 kHz (WavLM operates at 16 kHz; upsample mixture post-extraction)
- Mixture length: Up to 60 seconds (chunked internally for memory)
- TTS reference length: 3–30 seconds
- Format: Raw waveform, float32

### 1.2 Feature Extraction (Shared Encoder)

```
x(t) [24kHz] ──→ Downsample 16kHz ──→ WavLM-Large encoder [frozen initially]
                                              │
                                         H_mix ∈ R^{T×1024}  [50Hz frame rate]
                                              │
                                         Learned projection: Linear(1024→512)
                                              │
                                         X_mix ∈ R^{T×512}

ŝ(t) [24kHz] ──→ Downsample 16kHz ──→ WavLM-Large encoder [same frozen weights]
                                              │
                                         H_ref ∈ R^{T_ref×1024}
                                              │
                                         Learned projection: Linear(1024×512)
                                              │
                                         X_ref ∈ R^{T_ref×512}
```

**Why WavLM as feature extractor:**
- 2950 citations; best-known speaker representation (better than ECAPA for fine-grained similarity)
- Pre-trained on 94k hours of diverse speech — immediately multilingual-capable
- Layer-weighted combination trained jointly (like SUPERB); allows learning which layers are most informative for TSE
- 316M params but frozen during early training (finetune in Stage 3+)

### 1.3 CMHA Reference Conditioning Module

```
X_mix ∈ R^{T×512}    ←─── Query
X_ref ∈ R^{T_ref×512} ←─── Key, Value

For each CMHA block:
  Q = Linear(X_mix)   ∈ R^{T×512}
  K = Linear(X_ref)   ∈ R^{T_ref×512}
  V = Linear(X_ref)   ∈ R^{T_ref×512}

  # 8 heads, head dim = 64
  A = softmax(Q @ K^T / sqrt(64))  ∈ R^{T×T_ref}  [no causal mask]
  C = A @ V                         ∈ R^{T×512}    [frame-level ref features]

  # Reliability gate: suppress when reference is unreliable (noisy TTS)
  g = sigmoid(Linear([mean(C), std(C)]))  ∈ R^{512}  [per-dim gate]
  C_gated = g * C + (1-g) * zeros

X_fused = LayerNorm(X_mix + C_gated)  ∈ R^{T×512}
```

**Key properties:**
- T×T_ref attention — no positional alignment required. Handles TTS timing mismatch.
- Reliability gate: if TTS reference has artifacts or wrong speaker quality, gate suppresses contribution
- Applied at multiple depths in the separator (not just input)

### 1.4 Separation Backbone: Scaled TF-GridNet

```
X_fused ∈ R^{T×512}
   │
   ↓ STFT(window=25ms, hop=10ms) → Complex spectrogram C ∈ C^{T×F}
     [F = 257 bins at 16kHz, or 513 bins at 24kHz]
   │
   ↓ Band-Split (BSRNN-inspired): Split F into 12 log-spaced bands
     Band features: X_bands ∈ R^{T×B×D_band}  [B=12, D_band=128]
   │
   ┌─────────────────────────────────────────────────────────┐
   │  TF-GridNet Block × 18 (scaled from original 6)        │
   │                                                         │
   │  Per-block:                                            │
   │  1. CMHA(X_bands, X_ref) — inject reference per-band  │
   │     [CMHA shared weights across bands]                 │
   │  2. Intra-frame: Linear(band_dim → band_dim)           │
   │     + LayerNorm + GELU                                 │
   │  3. Sub-band temporal: Mamba (replaces BLSTM)          │
   │     [bidirectional Mamba, O(T) complexity]             │
   │  4. Full-band cross-attention: Q,K,V across all bands  │
   │     [captures harmonic relationships]                  │
   └─────────────────────────────────────────────────────────┘
   │
   ↓ Complex mask prediction:
     M_real, M_imag = Linear(X_bands_out) → R^{T×F}
     M_complex = tanh(M_real) + j·tanh(M_imag)  [bounded complex mask]
   │
   ↓ Apply to mixture STFT: Ĉ = M_complex ⊙ C_mix
   │
   ↓ iSTFT + overlap-add → ŝ_ext(t) at 24kHz
```

**Why Mamba instead of BLSTM:**
- O(T) vs O(T²) for bidirectional attention — handles 60-second audio
- SPMamba showed +2.42 dB SI-SNRi over TF-GridNet with Mamba modules
- Streaming-compatible recurrent form (unidirectional Mamba for streaming mode)
- 18 blocks instead of 6 — enabled by Mamba efficiency

**Why band-split:**
- Music contains harmonic energy distributed across specific frequency bands
- Band-specific processing + cross-band attention = music-aware separation
- BSRNN showed strong music source separation; we incorporate this into TSE backbone

### 1.5 Scale Plan

| Component | Params (Base) | Params (Large) |
|---|---|---|
| WavLM encoder (frozen initially) | 316M | 316M |
| WavLM projection | 0.5M | 1M |
| CMHA modules (×18 layers) | 28M | 56M |
| TF-GridNet backbone (18 Mamba blocks) | 180M | 360M |
| Band-split module | 8M | 16M |
| Complex mask head | 4M | 8M |
| **Total** | **~536M** | **~757M** |

**Track 1 target: ~750M params (no constraint)**

---

## TRACK 2: FLOW-MATCHING DiT WITH ENROLLMENT

This extends SAM Audio's architecture to support audio enrollment conditioning.

### 2.1 Overview

SAM Audio's DiT architecture is the foundation. We add an enrollment audio pathway using CMHA, parallel to the text/visual pathways.

```
Mixture x(t) ──→ DAC-VAE encoder ──→ z_mix ∈ R^{T'×128}
                                           │
TTS ref ŝ(t) ──→ WavLM ──→ CMHA ──→ z_ref ∈ R^{T'×512}
                                           │
              ┌─────────────────┐          │
              │  DiT Backbone   │ ←────────┘
              │  × 24 blocks    │
              │  Flow matching  │
              │  8 steps        │
              └────────┬────────┘
                       │
              DAC-VAE decoder ──→ ŝ_ext(t)
```

### 2.2 DiT Block with Enrollment Conditioning

```
Each DiT block:
  1. Self-attention on [noise_latent; z_mix] concat  [temporal context]
  2. Cross-attention to z_ref  [enrollment conditioning]
  3. Cross-attention to text embedding (optional, for domain control)
  4. FFN with SwiGLU activation
  5. AdaLN-Zero conditioning on timestep t
```

**Key difference from SAM Audio:** We add enrollment (z_ref) as an additional cross-attention pathway alongside text/visual. SAM Audio has no enrollment pathway — this is our contribution.

### 2.3 Training with Flow Matching

```
Flow matching objective:
  z_0 ~ N(0, I)  [noise]
  z_1 = clean audio latent (from DAC-VAE encoder on target speech)
  
  Interpolation: z_t = (1-t)*z_0 + t*z_1  for t ∈ [0,1]
  
  Model predicts: v_θ(z_t, t, x_mix_latent, x_ref_features)
  
  Loss: L = E_t[||v_θ - (z_1 - z_0)||²]
             [matches predicted to true velocity field]

Inference (8 steps):
  z = z_0 ~ N(0, I)
  for t in linspace(0, 1, 8):
    z += (1/8) * v_θ(z, t, x_mix_latent, x_ref_features)
  x_extracted = DAC-VAE decoder(z)
```

### 2.4 Scale Plan

| Component | Params |
|---|---|
| DAC-VAE (frozen) | ~74M |
| WavLM (frozen initially) | 316M |
| CMHA reference encoder | 45M |
| DiT backbone (24 blocks, dim=2048) | 2.1B |
| **Total** | **~2.5B** |

**Track 2 target: ~2.5B params**

---

## FULL FORWARD PASS DIAGRAM (Track 1)

```
                              TTS Synthesis
                      transcript + voice_clone(x(t))
                                    │
                                    ↓
x(t) ──────────────┐          ŝ(t) = TTS reference
[mixture, 24kHz]   │          [same words, approx voice]
                   │                │
                   ↓                ↓
           ┌──────────────────────────────┐
           │     WavLM-Large Encoder      │
           │  (shared weights, 16kHz)     │
           └──────┬──────────────┬────────┘
                  │              │
             H_mix∈R^{T×1024} H_ref∈R^{T_ref×1024}
                  │              │
           Proj(512)         Proj(512)
                  │              │
           X_mix∈R^{T×512}  X_ref∈R^{T_ref×512}
                  │              │
                  └──────┬───────┘
                         │
                    ┌────▼──────────────────────┐
                    │   CMHA Block              │
                    │   Q=X_mix, K=V=X_ref      │
                    │   + reliability gate       │
                    │   → X_fused∈R^{T×512}    │
                    └────────────┬──────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │   STFT + Band Split        │
                    │   12 log-spaced bands      │
                    └────────────┬──────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │  TF-GridNet × 18 blocks   │
                    │  (per block):             │
                    │  1. CMHA(X_bands, X_ref)  │
                    │  2. Intra-frame linear    │
                    │  3. Mamba (sub-band time) │
                    │  4. Full-band cross-attn  │
                    └────────────┬──────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │  Complex Mask             │
                    │  M_complex = tanh+j·tanh  │
                    └────────────┬──────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │  iSTFT + OLA              │
                    └────────────┬──────────────┘
                                 │
                            ŝ_ext(t)          [Optional: re-feed to
                        [extracted speech]     CMHA for 2nd pass]
```

---

## MULTILINGUAL HANDLING

**Approach: Shared encoder, language-balanced training**

WavLM-Large was pre-trained on English primarily, but it is the best available speaker representation model. We handle multilingual in three ways:

1. **Stage 3 fine-tuning of WavLM:** Fine-tune the top 8 layers of WavLM on Kathbath + Shrutilipi + MLS multilingual to adapt to Hindi. Full WavLM unfreezing in Stage 3+.

2. **Language-balanced batching:** At every step, batch contains 50% English, 30% Hindi, 10% code-switched, 10% other. WavLM adapts to Hindi spectral patterns within its pre-trained structure.

3. **No language-specific branches:** One shared model handles all languages. Language diversity in training is sufficient for WavLM to generalize.

**Expected Hindi performance gap:** ~2–3 dB SI-SDR below English initially, narrowing to ~1 dB after Stage 3 multilingual training (based on cross-lingual TSE literature).

---

## MUSIC AWARENESS

**Approach: Band-split + music-specific loss**

1. **Band-split frequency processing:** Explicitly handles harmonic structure. BSRNN showed strong music source separation. Our 12-band split (log-spaced from 50 Hz to Nyquist) covers all musical octaves.

2. **Music-specific auxiliary loss:** On music-mixed test conditions, add SDR loss on the music residual (encourage clean music separation alongside speech extraction):
   ```
   L_music = SDR(ŝ_ext(t) + music_residual(t), original_mixture_without_speech)
   ```
   This prevents the model from hallucinating speech artifacts into the music.

3. **HTDemucs-inspired full-band attention:** Cross-band attention in TF-GridNet captures harmonic relationships across octaves (e.g., F0 at 200 Hz and its 5th harmonic at 300 Hz).

4. **Training data:** 20% of batches include MusDB18-HQ music stems as interference (not just MUSAN). This forces the model to learn music-speech boundary separation.

---

## STREAMING VARIANT (Track 1-S)

For production deployment with latency constraints:

| Change | Impact |
|---|---|
| Bidirectional Mamba → Causal Mamba | ~1 dB SI-SDR reduction |
| 200ms chunks, 50ms overlap | Algorithmic latency = 200ms |
| Full-band cross-attn → linear attention | O(B) vs O(B²) |
| Pre-computed X_ref (static enrollment) | No per-chunk ref encoding |
| WavLM → smaller distilled version (WavLM-base) | 94M params, 4x faster |

**Track 1-S target parameters:** ~300M (WavLM-base + smaller backbone)  
**Target RTF:** < 0.1 on single A100, < 0.5 on CPU (with quantization)

---

## DESIGN DECISIONS AND JUSTIFICATIONS

| Decision | Why | Alternative Rejected |
|---|---|---|
| CMHA over pooled speaker embedding | Best WavLM Sim (0.935), timing-robust for TTS | ECAPA-TDNN pooling (loses timing info) |
| WavLM as feature extractor | 2950 citations; best speaker features; multilingual | Raw waveform encoder (worse for Hindi without large data) |
| Mamba over BLSTM | O(T) complexity, +2.42 dB (SPMamba), streaming-capable | BLSTM (memory O(T)) |
| Band-split over single-band | Music-aware harmonic processing | Single-band (blindly processes music) |
| Complex masking | Preserves phase, reduces music artifacts | Magnitude masking (phase reconstruction artifacts) |
| 18 blocks (scaled from 6) | Scale = quality; no compute constraint | 6 blocks (quality ceiling) |
| Flow-matching DiT as Track 2 | SAM Audio showed this scales to ~1M hrs | Diffusion (too slow, 200 steps) |
| DAC-VAE for Track 2 latent | SAM Audio uses it; compact representation | EnCodec (slightly lower quality) |
| 24 kHz output | Production quality; music requires >16kHz | 16 kHz (insufficient for music) |
