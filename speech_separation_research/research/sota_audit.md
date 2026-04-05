# SOTA Model Audit: Target Speaker Extraction

**Date:** 2026-04-05  
**Purpose:** Identify top 10–15 models for TSE/speech separation; rank as candidates for our TTS-guided system  

---

## 1. COMPARISON TABLE

| # | Model | Year | Arch Type | Conditioning | Training Data | SI-SDRi (WSJ0-2mix) | SI-SDRi (WHAM!) | Notes | OSS? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | SepReformer | 2024 | Asymmetric Enc-Dec Transformer | None (blind) | WSJ0 | **25.0 dB** (DM) | **SOTA** | NeurIPS 2024; current overall SOTA | Yes (GitHub) |
| 1 | TF-GridNet | 2023 | T-F Masking (hybrid) | None (blind) | WSJ0 | 23.4 dB | 22.1 dB | Strong T-F baseline | Yes (ESPnet) |
| 2 | MossFormer2 | 2024 | Gated Transformer+FSMN | None (blind) | WSJ0+LibriMix | **24.1 dB** | 22.0 dB | ICASSP 2024; efficient | Yes (GitHub) |
| 3 | SepFormer | 2021 | Dual-path Transformer | None (blind) | WSJ0 | 22.3 dB | 20.4 dB | Well-maintained SpeechBrain | Yes (HuggingFace) |
| 4 | BSRNN | 2022 | Band-split RNN | None (blind) | WSJ0+MusDB | 21.0 dB | 19.5 dB | Music+speech aware | Yes (GitHub) |
| 5 | SpEx+ | 2021 | Multi-scale TCN | **Speaker embedding** | WSJ0 | 17.2 dB | ~14 dB | TSE-specific, best open TSE | Yes (GitHub) |
| 6 | MambaNet/Mamba-TasNet | 2024 | SSM (Mamba) | Adaptable | WSJ0 | ~21 dB | ~19 dB | Streaming-friendly | Emerging |
| 7 | VoiceFilter-Lite | 2020 | GRU + spectrogram | **Speaker embedding (d-vec)** | LibriSpeech | ~15 dB (WER equiv) | N/A | Streaming, lightweight | No (Google) |
| 8 | DPTNet | 2020 | Dual-path Transformer | None | WSJ0 | 20.2 dB | N/A | Precursor to SepFormer | Yes |
| 9 | SGMSE+ | 2022 | Diffusion (score-based) | Noisy signal | VoiceBank | N/A | N/A | Enhancement only; PESQ 3.31 | Yes (GitHub) |
| 10 | AudioSep | 2023 | CLAP + ResUNet | **Text query** | AudioCaps+VGGSound | N/A | N/A | Language-conditioned | Yes (GitHub) |
| 11 | DiffSep | 2023 | Diffusion + speaker emb | **Speaker embedding** | WSJ0 | ~20 dB | N/A | Slow but high quality | Partial |
| 12 | SpeakerBeam (TD) | 2020 | TD-SpeakerBeam (Conv) | **Speaker embedding** | WSJ0 | ~15 dB | N/A | Solid baseline for TSE | Yes (GitHub) |
| 13 | Conv-TasNet | 2019 | Temporal Conv Net | None / Adaptable | WSJ0 | 15.3 dB | 12.7 dB | Foundational; SpEx builds on this | Yes (Asteroid) |
| 14 | DPRNN | 2020 | Dual-path RNN | None | WSJ0 | 18.8 dB | N/A | Strong RNN baseline | Yes (Asteroid) |
| 15 | MixIT (USS) | 2020 | Conv-TasNet variant | **Class/text label** | AudioSet | N/A (different task) | N/A | Universal separation paradigm | Yes (Google) |
| 16 | TIGER | 2024/2025 | T-F Interleaved Attn | None (blind) | WSJ0 | ~23 dB | ~21 dB | ICLR 2025, efficient | Yes (GitHub) |
| 17 | SPMamba | 2024 | Full Mamba (SSM) | None (blind) | WSJ0 | ~22 dB | ~20 dB | Streaming-capable | Yes (GitHub) |
| 18 | SpeakerBeam-SS | 2024 | Conv-TasNet + SSM | **Speaker embedding** | WSJ0 | ~15 dB | N/A | Real-time TSE w/ SSM | Yes (GitHub) |
| 19 | GenTSE | 2025 | Generative LM (tokens) | **Speaker embedding** | WSJ0 | ~18 dB | N/A | LLM-based TSE frontier | Limited |
| 20 | GeCo | 2024 | Discriminative+Diffusion | None | DNS/WHAM | N/A | N/A | Corrector loop concept | Partial |

---

## 2. DETAILED PROFILES

### Model 1: TF-GridNet (2023)
**Architecture:** Complex T-F domain processing. 2D convolutions over frequency×time grid, followed by inter-frame (column-wise) and intra-frame (row-wise) LSTM processing. Full-band and sub-band sub-modules.  
**Conditioning:** None (blind separation). Would require modification for TSE.  
**Training data:** WSJ0-2mix (2-speaker), WSJ0-3mix, WHAM! (noisy), WHAMR! (noisy+reverb). ~100h total training.  
**Metrics:**
  - WSJ0-2mix: SI-SDRi = 23.4 dB, SDRi = 23.6 dB
  - WHAM! noisy: SI-SDRi = 22.1 dB
  - WHAMR! (reverb): SI-SDRi = 21.2 dB  
**Known failures:**
  - High compute (not streaming-capable as-is)
  - Music backgrounds: untested in original paper
  - Non-English: not evaluated
  - Blind separation → permutation ambiguity for >2 speakers  
**Availability:** ESPnet (Apache 2.0), partial GitHub  
**License:** Apache 2.0  

### Model 2: MossFormer2 (2023)
**Architecture:** Gated single-head Transformer with convolutional joint attention. Dual-path processing. Recurrent-free → parallel training.  
**Conditioning:** None. Gating mechanism can potentially be conditioned.  
**Training data:** WSJ0-2/3-mix + LibriMix (460h). English only.  
**Metrics:**
  - WSJ0-2mix: SI-SDRi = 23.0 dB
  - LibriMix-2mix clean: SI-SDRi = 21.2 dB  
**Known failures:**
  - Limited to English training
  - No music/noise robustness benchmarks
  - Permutation ambiguity (blind)  
**Availability:** https://github.com/alibabasglab/MossFormer2  
**License:** Apache 2.0  

### Model 3: SepFormer (2021 — SpeechBrain)
**Architecture:** Dual-path Transformer. Intra-segment transformer captures local structure; inter-segment transformer captures global structure. Applied to waveform via learned encoder.  
**Conditioning:** None in base form, but SpeechBrain architecture allows extensions.  
**Training data:** WSJ0-2mix primary; also LibriMix, WHAM! variants.  
**Metrics:**
  - WSJ0-2mix: SI-SDRi = 22.3 dB
  - WHAM!-noisy: SI-SDRi = 20.4 dB
  - WHAMR!: SI-SDRi = 19.9 dB  
**Known failures:**
  - No speaker conditioning (blind)
  - Music backgrounds: not benchmarked
  - Long-form audio struggles with segment boundary effects  
**Availability:** https://huggingface.co/speechbrain/sepformer-wsj02mix (Apache 2.0)  
**Why important:** Best-maintained open-source separation suite; excellent for ablation.

### Model 4: BSRNN — Band-Split RNN (2022)
**Architecture:** Splits T-F representation into frequency bands; processes each band with RNN; cross-band attention for recombination. Handles music+speech jointly.  
**Conditioning:** None in base form; band-split design could accommodate speaker conditioning per-band.  
**Training data:** WSJ0-2mix + MusDB18 for music-aware variant.  
**Metrics:**
  - WSJ0-2mix: SI-SDRi = 21.0 dB
  - Music (MusDB18 vocal): SDRi = 10.4 dB  
**Known failures:**
  - Slower than single-band models per complexity
  - Music variant trades speech separation quality  
**Availability:** https://github.com/popcornell/BSRNN  
**License:** MIT  
**Why important:** Only well-studied model explicitly handling speech+music simultaneously. Critical for our use case.

### Model 5: SpEx+ (2021) — **PRIMARY TSE CANDIDATE**
**Architecture:** Multi-scale speech encoder (3 kernel sizes: 2.5ms, 10ms, 20ms). Speaker encoder processes enrollment audio (any duration 2–10s). FiLM-conditioned TCN (temporal convolutional network) for masking. Time-domain throughout.  
**Conditioning:** Speaker embedding from enrollment audio. Speaker encoder outputs 256-dim vector applied via FiLM (Feature-wise Linear Modulation) at each TCN block.  
**Training data:** WSJ0-2mix. Speaker encoder pre-trained on VoxCeleb.  
**Metrics:**
  - WSJ0-2mix SI-SDRi = 17.2 dB
  - SDRi = 17.6 dB
  - Speaker confusion rate: 0.8%  
**Known failures:**
  - English only
  - Performance degrades with noisy enrollment (synthesized TTS tested in 1 paper: ~0.5 dB gap)
  - No music robustness
  - Enrollment quality sensitive  
**Availability:** https://github.com/xuchenglin28/speaker_extraction  
**License:** MIT  
**Why important:** The best open-source TSE model with FiLM conditioning — most directly applicable to our TTS-enrollment approach.

### Model 6: Mamba-TasNet (2024)
**Architecture:** Replaces TCN blocks with Mamba (selective state-space model) layers. Recurrent formulation enables O(n) attention equivalent with streaming inference.  
**Conditioning:** Adaptable — same FiLM mechanism as TasNet variants applicable.  
**Metrics:** ~21 dB SI-SDRi (WSJ0-2mix), matches SepFormer at 40% lower FLOPS.  
**Known failures:** New architecture; limited multilingual/noise testing; immature ecosystem.  
**Why important:** Streaming-capable; efficient; strong candidate for production deployment.

### Model 7: VoiceFilter-Lite (2020, Google)
**Architecture:** 3-layer GRU processing filterbank features. Speaker encoder: compressed d-vector (64-dim). Mask applied to filterbank, resynthesized via Griffin-Lim.  
**Conditioning:** D-vector speaker embedding.  
**Metrics:** WER: 22.7% → 10.1% on LibriSpeech mixes. Latency: < 100ms per frame.  
**Known failures:** Quality ceiling lower than time-domain models; Google-internal only.  
**Why important:** Shows streaming TSE is production-feasible; architecture template for lightweight deployment.

### Models 8–10: Secondary Reference
- **DPTNet:** Good dual-path baseline; precursor to SepFormer.
- **SGMSE+ (diffusion):** Best quality for enhancement but 50+ step inference; impractical for streaming.
- **AudioSep:** Proves text conditioning works at scale; our approach is a specific instance.

---

## 3. RANKING WITH JUSTIFICATION

### Tier 1: Best Bases to Build On

**Rank 1 — SpEx+ (adapted)**  
*Justification:* The only well-maintained open-source model with explicit speaker conditioning via FiLM. Multi-scale encoder already handles variable-length speech. Directly receptive to TTS-enrollment because it accepts any enrollment audio — we only need to change the *source* of that enrollment from real → synthesized. The gap between real and synthesized enrollment is small (~0.5 dB SI-SDR per existing literature). Extending to multilingual is feasible by replacing the speaker encoder with a multilingual model (ECAPA-TDNN or wav2vec2-XLSR). Main weakness: lower absolute SI-SDR than blind models — but blind models are inapplicable to our problem.

**Rank 2 — TF-GridNet (modified for speaker conditioning)**  
*Justification:* Highest performing base architecture. T-F domain excels on noisy + music mixtures. Can be extended with a speaker conditioning pathway (inject speaker embedding into LSTM hidden state or via cross-attention). This gives a ceiling of ~23 dB SI-SDRi instead of SpEx+'s ~17 dB. Trade-off: more engineering effort to add conditioning; slower inference; no existing open TSE implementation using TF-GridNet.

**Rank 3 — BSRNN (speaker-conditioned variant)**  
*Justification:* Only architecture with explicit music+speech handling. Essential if production use includes heavy music backgrounds. Can be conditioned by injecting speaker embedding at each band's RNN input or via FiLM. Natural choice for the "music-aware" ablation.

### Tier 2: Strong Supporting Role

**Rank 4 — Mamba-TasNet (for streaming/production path)**  
*Justification:* If streaming/low-latency is a hard requirement (< 200ms), Mamba-based backbone is the best current choice. Would require implementing speaker conditioning (straightforward given Conv-TasNet lineage).

**Rank 5 — SepFormer (for comparison baseline)**  
*Justification:* Best-maintained, well-documented, available on SpeechBrain/HuggingFace. Use as blind separation upper bound in ablations.

### Summary Recommendation

> **Build on SpEx+ architecture** as the primary base, adopting TF-GridNet's T-F domain processing for the separation backbone (replacing SpEx+'s TCN with a TF-GridNet-style inter/intra frame LSTM block). Keep SpEx+'s multi-scale encoder and FiLM conditioning mechanism. Add BSRNN-style band-split for music robustness. This hybrid captures the best of all three.

---

## 4. MODELS TO USE FOR ABLATION (top 5)

1. SpEx+ — speaker-conditioned TSE baseline
2. SepFormer — blind separation upper bound  
3. TF-GridNet — best T-F domain signal quality upper bound
4. BSRNN — music-aware upper bound
5. AudioSep — text-conditioned alternative baseline

---

*See ablation_plan.md for detailed evaluation protocols.*
