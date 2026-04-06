# Ablation Plan v2: TTSE-Net Systematic Ablations
**Date:** 2026-04-06  
**Based on:** architecture_design_v2.md, training_plan_v2.md, gap_analysis_v2.md

---

## OVERVIEW

The ablation plan answers the five design questions that drive our architecture choices. Each ablation is self-contained: one variable changed, all others held constant. Ablations are run on a fixed 20% sub-sample of LibriMix-2mix-clean + MusicMix-TSE-0dB to control compute cost.

**Standard evaluation set for all ablations:**
- LibriMix-2mix-clean test: 500 utterances (oracle enrollment unless otherwise noted)
- MusicMix-TSE-0dB: 200 utterances
- TTS enrollment condition: 200 utterances (LibriMix, CosyVoice 2 clone)
- Primary metric: SI-SDRi (librimix), WavLM-Sim (tts condition)

---

## ABLATION SERIES A: CONDITIONING MECHANISM

### Research Question
How does the method of incorporating the TTS enrollment signal affect extraction quality, and specifically: does CMHA's timing invariance provide a measurable advantage when the reference is synthesized?

### A.1: Conditioning Method Comparison

Train 5 identical models (TF-GridNet backbone, 120M params, Stage 2 conditions) with only the conditioning mechanism varied:

| Model | Conditioning | Description |
|-------|-------------|-------------|
| A1-a | **No conditioning** | Blind separation baseline (beamformer) |
| A1-b | **d-vector FiLM** | ECAPA-TDNN 192-d embedding, FiLM layers |
| A1-c | **WavLM-mean FiLM** | WavLM-Large mean pooled, FiLM layers |
| A1-d | **WavLM-mean concat** | WavLM-Large mean pooled, concat to each frame |
| A1-e | **CMHA (ours)** | CMHA over full WavLM sequence, no pooling |

**Enrollment conditions evaluated for each:**
- Oracle real speech
- TTS voice clone (CosyVoice 2)
- TTS wrong timing (0.6× speed)
- TTS generic voice (no cloning)

**Expected outcome:**
- A1-e > A1-c > A1-b > A1-d > A1-a on TTS enrollment (timing-robustness)
- A1-e ≈ A1-c on oracle enrollment (less benefit of CMHA when timing matches)
- Delta between oracle and TTS enrollment: A1-e < A1-c (CMHA closes the gap)

### A.2: Number of CMHA Heads

| Model | CMHA Heads | Params added |
|-------|-----------|-------------|
| A2-a | 1 head | Minimal |
| A2-b | 4 heads | Small |
| A2-c | 8 heads (default) | Medium |
| A2-d | 16 heads | Large |
| A2-e | 32 heads | Very large |

Expected: diminishing returns past 8 heads; 16 might help for music (richer spectral attention).

### A.3: CMHA Injection Depth

| Model | CMHA Position | Description |
|-------|--------------|-------------|
| A3-a | Input only | Fuse before all TF-GridNet blocks |
| A3-b | Every 3 blocks | Inject reference at blocks 3,6,9,12,15,18 |
| A3-c | Every block (default) | Full injection at all 18 blocks |
| A3-d | Last 6 blocks only | Late fusion |
| A3-e | Middle 6 blocks only | Mid-network fusion |

Expected: Every-block injection (A3-c) best for speaker identity; input-only (A3-a) simplest.

### A.4: Reliability Gate for TTS Reference

The reliability gate predicts whether the TTS enrollment is trustworthy enough to use:

| Model | Gate Type | Description |
|-------|----------|-------------|
| A4-a | No gate | Always use TTS enrollment at full weight |
| A4-b | Fixed threshold | Use if WavLM-Sim(TTS, reference_pool) > 0.7 |
| A4-c | Soft gate (learned) | Sigmoid weight on CMHA output |
| A4-d | Hard gate (learned) | Binary decision from small classifier |
| A4-e | Confidence annealing | Gradually reduce TTS weight from 1→0 as quality drops |

Expected: A4-c or A4-e best — gradual degradation is better than hard switching.

---

## ABLATION SERIES B: BACKBONE ARCHITECTURE

### Research Question
Does the Mamba temporal module provide a material improvement over BLSTM, and does the band-split module help for music?

### B.1: Temporal Module

| Model | Temporal Module | Params | Complexity |
|-------|----------------|--------|-----------|
| B1-a | BLSTM (original TF-GridNet) | ~120M | O(T²) |
| B1-b | Transformer + RoPE | ~130M | O(T²) |
| B1-c | Mamba-1 (SSM) | ~120M | O(T) |
| B1-d | Mamba-2 (SSD) | ~125M | O(T) |
| B1-e | **Mamba-2 (default)** | ~125M | O(T) |
| B1-f | Hybrid: Mamba-2 + full-band cross-attn | ~135M | O(T) + O(B²) |

**Evaluation conditions:** Standard + 60s long-form (where O(T) complexity matters most)

Expected: Mamba-2 ≈ BLSTM for 5s utterances; Mamba-2 >> BLSTM for 60s utterances.

### B.2: Frequency Processing

| Model | Freq Processing | Description |
|-------|----------------|-------------|
| B2-a | Global STFT | Single mask over all frequencies |
| B2-b | Band-split (4 bands) | Coarse split |
| B2-c | Band-split (8 bands) | Medium split |
| B2-d | Band-split (12 bands) | Default (BSRNN-inspired) |
| B2-e | Band-split (24 bands) | Fine split, close to mel-filterbank |
| B2-f | Learned filterbank | Learnable frequency groupings |

**Key evaluation:** Music residual metric — does band-split reduce music bleed-through?

Expected: B2-d or B2-e best on music conditions; diminishing returns beyond 12 bands for speech-only.

### B.3: STFT Resolution

| Model | FFT Size | Hop Size | Time Resolution | Freq Resolution |
|-------|---------|---------|----------------|----------------|
| B3-a | 512 | 128 | 5.3ms | 46.9 Hz |
| B3-b | 1024 | 256 | 10.7ms | 23.4 Hz |
| B3-c | 2048 | 512 (default) | 21.3ms | 11.7 Hz |
| B3-d | 4096 | 1024 | 42.7ms | 5.9 Hz |
| B3-e | Multi-resolution | 512+1024+2048 | Multiscale | Multiscale |

Expected: Multi-resolution (B3-e) best for music; 2048 (B3-c) best for speech-only.

### B.4: Model Scale

| Model | Params | Architecture | Training |
|-------|--------|-------------|---------|
| B4-a | 34M | TF-GridNet-6blocks + CMHA | Stage 1-2 |
| B4-b | 120M | TF-GridNet-12blocks + CMHA | Stage 1-2 |
| B4-c | 250M | TF-GridNet-18blocks + Mamba | Stage 1-2 |
| B4-d | 500M | TF-GridNet-24blocks + Mamba | Stage 1-2 |
| B4-e | 750M (Track 1) | TF-GridNet-18blocks + Mamba + wider | Stage 1-5 |

Expected: Steady improvement 34M → 750M on SI-SDR; larger benefit on TTS enrollment condition.

---

## ABLATION SERIES C: TRAINING DATA

### Research Question
What data and training curriculum contribute most to generalization?

### C.1: TTS Enrollment Percentage During Training

| Model | TTS % in Training | Real % |
|-------|-----------------|--------|
| C1-a | 0% TTS | 100% real enrollment |
| C1-b | 10% TTS | 90% real |
| C1-c | 30% TTS (Stage 2 target) | 70% real |
| C1-d | 60% TTS (Stage 3-4 target) | 40% real |
| C1-e | 80% TTS (Stage 5 target) | 20% real |
| C1-f | 100% TTS | 0% real |

**Key evaluation:** SI-SDR gap (oracle real − TTS enrollment)  
Expected: 60–80% TTS in training minimizes gap; 100% may hurt oracle condition.

### C.2: TTS System Diversity During Training

| Model | TTS Systems in Training | Expected Benefit |
|-------|------------------------|-----------------|
| C2-a | CosyVoice 2 only | Overfit to one TTS style |
| C2-b | CosyVoice 2 + F5-TTS | Moderate diversity |
| C2-c | All 3 (CosyVoice 2 + F5-TTS + XTTS) | Default |
| C2-d | All 3 + speed perturbation | Timing invariance |
| C2-e | All 3 + speed + pitch + formant | Full augmentation |

Expected: C2-d and C2-e significantly outperform C2-a on unseen TTS systems.

### C.3: Music Training Data Volume

| Model | Music Training Hours | Source |
|-------|--------------------|----|
| C3-a | 0h (no music) | Baseline |
| C3-b | 500h | MusDB18 + FMA-small |
| C3-c | 2,000h | + MedleyDB |
| C3-d | 5,000h | + OpenSoundscape |
| C3-e | 14,000h (default) | + MUSDB18-HQ + FMA-full |

**Key evaluation:** Music residual metric on MusicMix-TSE  
Expected: Sharp improvement from 0→500h; plateau beyond 5,000h.

### C.4: Hindi Data Volume

| Model | Hindi Training Hours | Source |
|-------|--------------------|----|
| C4-a | 0h (English only) | Baseline |
| C4-b | 200h | Kathbath |
| C4-c | 1,000h | + Shrutilipi |
| C4-d | 5,000h | + EMILIA-Hindi |
| C4-e | Full multilingual (default) | All sources |

**Key evaluation:** Hindi-2mix SI-SDR, WavLM-Sim (cross-lingual)  
Expected: Even 200h provides significant improvement; gains continue to 5,000h.

### C.5: Dynamic Mixing vs. Static

| Model | Mixing Strategy | Training Complexity |
|-------|---------------|-------------------|
| C5-a | Static mixtures | Fixed dataset, no variation |
| C5-b | Dynamic mixing only | On-the-fly, new mix every step |
| C5-c | 50% static + 50% dynamic | Hybrid |
| C5-d | Dynamic + RIR aug | + random room acoustics |
| C5-e | Dynamic + RIR + music aug (default) | Full augmentation |

Expected: C5-e best generalization; C5-a worst (overfits to specific mixing conditions).

---

## ABLATION SERIES D: LOSS FUNCTION

### Research Question
Does combining SI-SDR with MRSTFT, speaker similarity, and music-aware losses improve results?

### D.1: Loss Component Ablation

| Model | Losses Active | Formula |
|-------|-------------|---------|
| D1-a | SI-SDR only | L = SI-SDR |
| D1-b | + MRSTFT | L = SI-SDR + λ₁ MRSTFT |
| D1-c | + MRSTFT + SpeakerSim | L = SI-SDR + λ₁ MRSTFT + λ₂ SpkSim |
| D1-d | + MRSTFT + SpeakerSim + MusicResidual | Default |
| D1-e | + all + CTC (Stage 5) | Full loss schedule |

**Key metrics:** PESQ (perceptual), WavLM-Sim (speaker), Music Residual (music), SI-SDR  
Expected: Each loss term improves its corresponding metric; SI-SDR may dip slightly with full loss.

### D.2: Loss Weights

| Model | λ_MRSTFT | λ_SpkSim | λ_MusicRes |
|-------|---------|---------|-----------|
| D2-a | 0.1 | 0.1 | 0.1 |
| D2-b | 0.5 | 0.1 | 0.1 |
| D2-c | 0.5 | 0.5 | 0.1 |
| D2-d | 0.5 | 0.5 | 0.5 (default) |
| D2-e | 1.0 | 1.0 | 1.0 |
| D2-f | 1.0 | 0.5 | 0.2 |

Expected: D2-d or D2-f best balance; high SpkSim weight may hurt SI-SDR.

### D.3: DPO Alignment (Stage 5)

| Model | DPO Setup | Reference Model |
|-------|----------|----------------|
| D3-a | No DPO | Stage 4 checkpoint |
| D3-b | DPO on WavLM-Sim preference pairs | WavLM similarity > threshold |
| D3-c | DPO on DNSMOS preference pairs | DNSMOS > threshold |
| D3-d | DPO on SI-SDR preference pairs | Higher SI-SDR = preferred |
| D3-e | DPO multi-objective (SIM + DNSMOS) | Both criteria |

Expected: D3-e best overall; D3-b best for TTS enrollment (speaker identity most important).

---

## ABLATION SERIES E: ITERATIVE REFINEMENT

### Research Question
How many refinement passes are needed, and does TTS enrollment converge to oracle performance?

### E.1: Refinement Pass Count

| Pass | Enrollment Source | Expected SI-SDRi | Expected WavLM-Sim |
|------|-----------------|-----------------|------------------|
| 0 | TTS synthesis | Baseline | Baseline |
| 1 | ŝ₁ (output of pass 0) | +1–2 dB | +0.05–0.1 |
| 2 | ŝ₂ (output of pass 1) | +0.5–1 dB | +0.02–0.05 |
| 3 | ŝ₃ (output of pass 2) | +0.1–0.3 dB | +0.01 |
| Oracle | Real speech | Upper bound | Upper bound |

**Stopping criterion:** ΔSI-SDR < 0.2 dB between consecutive passes

### E.2: Starting Point Comparison

| Model | Pass 0 Enrollment | Pass 1 Enrollment |
|-------|-----------------|-----------------|
| E2-a | Oracle real speech | Same oracle |
| E2-b | TTS voice clone | Output of pass 0 |
| E2-c | TTS generic voice | Output of pass 0 |
| E2-d | TTS wrong timing (0.6×) | Output of pass 0 |
| E2-e | Silence (no enrollment) | Output of pass 0 |

Key question: Does iterative refinement converge to the same endpoint regardless of starting enrollment quality?

### E.3: Refinement with Reliability Gate

| Model | Gate Applied at | Expected Behavior |
|-------|---------------|--------------------|
| E3-a | No gate | Always refine |
| E3-b | Gate at pass 0 only | Suppress bad TTS, then refine |
| E3-c | Gate at every pass | Adaptively weight each pass |
| E3-d | Gate + early stopping | Stop if gain < threshold |

---

## ABLATION SERIES F: TRACK 2 — FLOW MATCHING

### Research Question
For the generative Track 2 model, how do diffusion steps, conditioning strength, and DiT depth affect perceptual quality?

### F.1: Number of Flow Matching Steps

| Model | Steps | Inference RTF | DNSMOS |
|-------|-------|-------------|--------|
| F1-a | 1 step (MeanFlow distilled) | ~0.05 | TBD |
| F1-b | 4 steps | ~0.1 | TBD |
| F1-c | 8 steps (default) | ~0.2 | TBD |
| F1-d | 16 steps | ~0.4 | TBD |
| F1-e | 32 steps | ~0.8 | TBD |

Expected: Quality plateau around 8 steps; 1-step MeanFlow significantly worse but faster.

### F.2: CMHA Conditioning Strength (Classifier-Free Guidance)

| Model | CFG Scale w | Behavior |
|-------|-----------|---------|
| F2-a | w = 0 | Unconditional (no speaker info) |
| F2-b | w = 1 | Light conditioning |
| F2-c | w = 2.5 (default) | Medium conditioning |
| F2-d | w = 5 | Strong conditioning |
| F2-e | w = 10 | Very strong (may over-constrain) |

Expected: w=2.5–5 best for WavLM-Sim; w>5 may hurt DNSMOS (over-fit to reference artifacts).

### F.3: DiT Depth vs. Width

| Model | Layers | Hidden dim | Params |
|-------|--------|-----------|--------|
| F3-a | 12 | 512 | ~200M |
| F3-b | 16 | 768 | ~600M |
| F3-c | 24 | 1024 (default) | ~2.5B |
| F3-d | 32 | 1024 | ~3.3B |
| F3-e | 24 | 1536 | ~5.6B |

Expected: F3-c (default) best cost-quality tradeoff; F3-d or F3-e may improve music.

---

## ABLATION EXECUTION SCHEDULE

### Phase 1: Core conditioning ablations (Weeks 10–14)
- Series A: Conditioning mechanism — 20 model variants
- Compute: 2 × H100, 3 days each = ~60 GPU-days
- **Decision gate:** Choose conditioning method for all subsequent ablations

### Phase 2: Architecture ablations (Weeks 14–18)
- Series B: Backbone — 25 model variants
- Compute: ~80 GPU-days
- **Decision gate:** Finalize backbone for full training

### Phase 3: Training strategy ablations (Weeks 18–22)
- Series C + D: Data and loss — 25 model variants
- Compute: ~100 GPU-days
- **Decision gate:** Finalize training recipe

### Phase 4: Refinement and DPO (Weeks 28–32)
- Series E + F (partial): Iterative refinement + Track 2 steps
- Compute: ~60 GPU-days

### Total ablation compute
- ~300 GPU-days × H100 = 300 × $2.50/hr × 24hr ≈ $18,000
- This is ~25% of total project compute budget

---

## RESULT REPORTING FORMAT

Each ablation series produces:
1. **Table:** Model × metric grid (SI-SDR, WavLM-Sim, DNSMOS, Music Residual)
2. **Plot:** Primary metric vs. swept variable (e.g., CMHA heads vs. SI-SDR)
3. **Oracle delta:** (oracle enrollment − TTS enrollment) for each model — our key novel metric
4. **Selection rationale:** One paragraph explaining which variant was selected and why

### Statistical significance
- All results reported as mean ± std over 5 random seeds for final ablations
- For preliminary ablations: single seed, but note this
- Significance test: paired t-test, p < 0.05 threshold

### Expected Final Numbers (from approach_comparison_v2.md baselines)

| Model | SI-SDRi (LibriMix) | WavLM-Sim (TTS enroll) | DNSMOS |
|-------|-------------------|----------------------|--------|
| SpEx+ (baseline) | ~17 dB | ~0.878 | ~3.186 |
| USEF-TSE (baseline) | ~20 dB | ~0.935 | ~3.272 |
| TTSE-Net-Track1 (target) | **>22 dB** | **>0.93** | **>3.3** |
| TTSE-Net-Track2 (target) | N/A (gen) | **>0.93** | **>3.6** |
| Ablation gap (oracle−TTS) | <2 dB | <0.03 | <0.1 |
