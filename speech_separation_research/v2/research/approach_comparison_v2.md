# Approach Comparison: Five Paradigms for TTS-Guided TSE
**Date:** 2026-04-06  
**Based on:** PDF extraction, real benchmark numbers, challenge results  

---

## THE FIVE PARADIGMS

| # | Paradigm | Core Idea | Best Model | Best SI-SDR | Perceptual | Speaker Sim | Inference Speed | Multilingual |
|---|---|---|---|---|---|---|---|---|
| A | Discriminative Masking | T-F or waveform mask | SepReformer | 25.4 dB | Moderate | Moderate | Very Fast | Trainable |
| B | Embedding-Free Discriminative | CMHA on full reference | USEF-TFGridNet | 23.3 dB (noisy+rev) | Moderate | **Best (0.935)** | Fast | Trainable |
| C | Masked Generative (codec) | MaskGIT on audio tokens | AnyEnhance | N/A (gen.) | **Best (3.638 DNSMOS)** | High | Medium | Trainable |
| D | AR Token LM | LM over codec tokens | LauraTSE | N/A (gen.) | Very High | High (0.908) | Slow | Potentially |
| E | Flow Matching | DiT + rectified flow | SAM Audio / Geneses | N/A (gen.) | Highest | TBD | Fast (1-10 steps) | Yes (SAM) |

---

## PARADIGM A: Discriminative Masking

### How it works
Given mixture spectrogram (or waveform encoding) and speaker conditioning signal, predict a mask over T-F bins or waveform frames. Apply mask to recover target speech.

### Best current model: SepReformer (NeurIPS 2024)
- 25.4 dB SI-SNRi on WSJ0-2mix with dynamic mixing
- Asymmetric encoder-decoder, no chunking bottleneck
- Speaker conditioning: add speaker embedding via FiLM or cross-attention

### Speaker conditioning variants in this paradigm
- **FiLM (SpEx+):** γ, β vectors from speaker embedding applied to each layer. Good but pooled embedding loses fine-grained structure.
- **Cross-attention to reference (USEF-TSE):** Better than FiLM — uses full reference sequence, not just pooled embedding.
- **Concatenation (VoiceFilter):** Speaker embedding concatenated to every frame of mixture features. Simple but effective.

### Strengths for our use case
- Highest SI-SDR scores
- Deterministic output — same input always gives same output (important for production)
- Fast inference (single forward pass)
- Well-understood failure modes
- Easiest to integrate with downstream ASR (Whisper)

### Weaknesses
- May over-smooth at very low SNR or heavy reverberation
- Artifacts at music-speech boundaries
- SI-SDR optimization ≠ perceptual quality optimization
- Speaker conditioning via pooled embedding loses fine timing cues from TTS reference

### For TTS enrollment: compatibility
- **Moderate.** Standard FiLM conditioning: TTS embedding works (< 0.5 dB SI-SDR gap per literature)
- **Best variant:** USEF-style CMHA — cross-attends to full TTS audio sequence, not pooled embedding. Handles timing mismatch naturally.

---

## PARADIGM B: Embedding-Free Discriminative (USEF-TSE)

### How it works
No speaker encoder. Cross Multi-Head Attention (CMHA) takes mixture encoding as query, full reference audio encoding as key/value. Produces frame-level features maintaining mixture sequence length. These fused features enter a discriminative separator.

### Why this is best for TTS enrollment

The fundamental problem with TTS enrollment:
- TTS-synthesized speech has correct speaker identity but wrong timing/prosody
- A pooled embedding (ECAPA, WavLM mean-pool) captures speaker identity ✓ but ignores temporal structure ✓
- CMHA captures speaker identity AND fine-grained spectral patterns from all reference frames simultaneously
- Because CMHA is permutation-flexible (attention over time), it doesn't care that the reference has wrong timing

**Concrete example:** TTS says "hello world" in 1 second, real speech says "hello world" in 0.7 seconds. A pooled embedding ignores this. CMHA attends to wherever in the TTS reference each spectral pattern appears — irrelevant that it's at wrong timestamp.

### Results (from LauraTSE comparison table)
- WavLM Sim: **0.935** (best among all models compared)
- WeSpeaker Sim: **0.988** (best)
- DNSMOS OVL: 3.272 (slightly below generative models)
- dWER: 4.319 (good, close to SpEx+)
- Training time slower due to reference encoding at every step, but inference is single pass

### Weaknesses
- Reference must be encoded at full sequence length (memory proportional to reference duration)
- DNSMOS below AnyEnhance / LauraTSE (no generative quality boost)
- Not yet tested at scale (1B+ params)

---

## PARADIGM C: Masked Generative (AnyEnhance Style)

### How it works
Encode audio as discrete tokens via neural codec (DAC, EnCodec, WavTokenizer). Use a MaskGIT-style masked generative model: iteratively unmask tokens conditioned on a prompt (reference audio) and partially observed mixture. HiFi-GAN or codec decoder reconstructs final audio.

### Why AnyEnhance is notable
- **Prompt-guidance:** At inference, provide a reference audio "prompt" — the model uses it for in-context learning (no fine-tuning). If the prompt is TTS-synthesized, it still works.
- **One model, all tasks:** Denoising, dereverberation, declipping, super-resolution, TSE — no task-specific heads.
- **Training data:** ~21k hours from Emilia (DNSMOS>3.4 filtered) + multi-source noise/RIR.
- **DNSMOS: 3.638** — best perceptual quality in the comparison table.

### Strengths for our use case
- Prompt-guidance is directly applicable to TTS enrollment (give TTS audio as prompt)
- Best perceptual quality and naturalness
- Non-autoregressive (MaskGIT) — faster than LM-based approaches
- Handles diverse degradations in one model (practical for production)
- Self-critic improves quality iteratively without retraining

### Weaknesses  
- No SI-SDR reported (generative — metric mismatch)
- Speaker similarity (WavLM Sim=0.735) lower than USEF-TSE (0.935) or LauraTSE (0.908)
- Less intelligibility control (codec token space vs. waveform)
- Less interpretable failure modes
- Music-specific training not explored

---

## PARADIGM D: Autoregressive Token LM (LauraTSE / GenTSE)

### How it works
Encode mixture and reference as continuous embeddings (WavLM or WavTokenizer). An autoregressive LM generates codec tokens of the clean target speech one by one (or in parallel with a fast decoder). HiFi-GAN or codec decoder synthesizes final audio.

### Two-stage approach (GenTSE, LauraTSE)
- Stage 1 (LM): coarse semantic tokens — captures "what was said"
- Stage 2 (encoder-only LM): fine acoustic tokens — captures "how it sounds"
- Enables separate optimization of intelligibility (stage 1) and quality (stage 2)

### Best results (from LauraTSE PDF, real numbers)
- DNSMOS OVL: **3.609** (second-best after AnyEnhance 3.638)
- WavLM Sim: **0.908** (best among generative; second overall after USEF-TSE 0.935)
- dWER: **4.333** (best among all generative models, comparable to discriminative)
- LauraTSE-streaming: DNSMOS=3.596, WavLM Sim=0.897 — **real-time capable**

### DPO alignment (GenTSE)
- After training, use DPO to align outputs with human perceptual preferences
- This is a key differentiator vs. purely loss-based training
- dWER improved 0.217→0.172 with Frozen-LM Conditioning + DPO

### Strengths for our use case
- Naturally handles timing/prosody mismatch: the LM generates clean tokens from continuous embeddings, not from aligned references
- DPO alignment can be applied with Hindi/music preference data
- Streaming variant exists
- Codec space enables language-model-scale training

### Weaknesses
- Autoregressive → slow inference (token by token)
- More complex training (two stages + DPO)
- Speaker similarity still below USEF-TSE in the comparison table
- No published results on music or multilingual audio

---

## PARADIGM E: Flow Matching (SAM Audio / Geneses / FlowTSE)

### How it works
Learn a vector field that transports a Gaussian noise distribution to the target clean audio distribution in continuous latent space (VAE or raw). Rectified flow matching enforces straight ODE trajectories → very few inference steps (1–10 vs. 200 for diffusion).

### SAM Audio (Meta FAIR, Dec 2025) — The Landmark Model
- DiT architecture (same as Stable Diffusion 3, FLUX)
- Flow matching in DAC-VAE latent space
- Three conditioning types: text (T5), visual (SAM2 masks), temporal spans
- ~1M hours training: speech + music + sound effects
- **Outperforms all specialist models on SAM-Audio-Bench** (in-the-wild evaluation)
- **But: no audio enrollment support.** This is a fundamental design choice.

### Geneses — Closest to our use case
- Unified flow matching for enhancement + separation
- SSL features from noisy mixture condition the DiT
- Would need adaptation to accept audio enrollment via cross-attention
- DNSMOS/NISQA close to ground truth

### FlowTSE (arXiv 2505.14465)
- First flow-based TSE with speaker enrollment
- Mixture + reference audio condition the flow predictor
- MeanFlow distillation → 1-step inference

### Why flow matching is the frontier
1. Faster than diffusion (10 steps → quality of 200 diffusion steps)
2. Continuous latent → no codec quantization artifacts
3. DiT scales well with parameters (proven at 3B+ in vision)
4. SAM Audio shows ~1M hours is sufficient for generalization to in-the-wild
5. **One-step variants** (MeanFlow-TSE, AlphaFlowTSE) are approaching real-time

### Weaknesses
- Very new (2024–2025); not production-hardened
- No SI-SDR comparison possible (different metric space)
- Speaker identity preservation with enrollment needs validation
- Need large amounts of training data for flow matching to work well
- SAM Audio doesn't support enrollment — FlowTSE is much smaller

---

## RECOMMENDATION FOR OUR SYSTEM

Given our constraints and goals (no size/compute limits, train from scratch, TTS-guided, multilingual, music-aware), the right approach is a **Disc-Gen hybrid with CMHA conditioning**, structured as two training tracks:

### Track 1: Production/Deployment (Start here)
**Discriminative USEF-TFGridNet at scale (1B params)**
- CMHA conditioning on TTS reference (no pooled speaker encoder)
- TF-GridNet backbone for music+noise robustness
- Band-split frequency processing (BSRNN-inspired)
- Multilingual via WavLM features for both mixture and reference
- Target: >20 dB SI-SDR + highest speaker similarity

This gives the best combination of: speaker identity preservation, intelligibility, and inference speed. CMHA handles the TTS timing/prosody mismatch problem naturally.

### Track 2: Quality Ceiling (Research)  
**Flow-matching DiT conditioned on TTS reference (3B params, SAM-Audio-style)**
- Add audio enrollment conditioning to SAM Audio's architecture
- Replace text/visual prompts with CMHA-based audio prompt conditioning
- Flow matching for perceptual quality
- Target: SAM Audio quality + enrollment-based speaker control

This bridges the gap between SAM Audio (no enrollment) and FlowTSE (small scale).

### For TTS Enrollment: Why CMHA wins over pooled embeddings
- Timing-robust: attention doesn't require temporal alignment
- Fine-grained: attends to spectral patterns at any time position in reference
- No information bottleneck: 192-dim embedding vs. full sequence representation
- Composition: multiple TTS systems produce different timing but similar CMHA outputs
