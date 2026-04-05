# FINAL RESEARCH PLAN: TTS-Guided Target Speaker Extraction

**System Name:** TTSE-Net (TTS-guided Target Speaker Extraction Network)  
**Date:** 2026-04-05  
**Version:** 1.0  
**Status:** Ready for engineering handoff (pending ablation phase validation)  

---

## SECTION 1: EXECUTIVE SUMMARY

### Problem
We need to extract a specific person's voice from a complex audio mixture containing music, ambient noise, and other speakers. The only information we have about the target speaker (beyond the mixed audio) is: (a) an ASR-generated transcript of what they said, and (b) potentially a short voice sample. We do not have clean reference audio of the target speaker from a prior session.

### Our Solution
**Input chain:** Mixed audio → ASR → Transcript + voice cloning → TTS synthesis → conditioning signal → neural extraction model → clean target speech

We synthesize an approximation `ŝ(t)` of the target speaker's voice using zero-shot TTS and voice cloning. This synthesis has the right words but wrong timing/prosody. We use this as a conditioning signal for a custom neural separation model (TTSE-Net) that extracts the target speaker's voice from the mixture.

### Why This Is Novel
No existing TSE system is designed around a synthesized reference as its primary conditioning source. The gap is both practical (a real deployment scenario) and scientific (studying the impact of synthesis quality on extraction accuracy). Additionally, no published work addresses TSE for Hindi or Hindi-English code-switched audio — creating an entirely new benchmark contribution.

### Key Deliverables
1. **TTSE-Net model:** ~34M parameter speaker-conditioned separation model with music/noise awareness
2. **Multilingual TSE benchmark:** ~5,800 test samples across English, Hindi, and code-switched conditions
3. **Ablation study:** 5 SOTA models on new benchmark
4. **Production deployment path:** Streaming variant with < 200ms latency

### Business Impact
- Enables production deployment of speaker extraction where clean enrollment audio is unavailable
- Handles Indian language content (Hindi, code-switched) — critical for Indian market
- Handles music backgrounds (podcast, broadcast, live event scenarios)

---

## SECTION 2: LITERATURE CONTEXT

### 2.1 State of the Field

The field of target speaker extraction (TSE) has advanced rapidly since 2019. Key landmarks:

| Era | Key Work | SI-SDRi | Method |
|---|---|---|---|
| 2019 | VoiceFilter (Google) | ~12 dB | LSTM + d-vector |
| 2019 | Conv-TasNet | 15.3 dB | TCN (blind) |
| 2020 | SpEx+ | 17.2 dB | Multi-scale TCN + FiLM |
| 2021 | SepFormer | 22.3 dB | Dual-path Transformer (blind) |
| 2022 | BSRNN | 21.0 dB | Band-split RNN (music+speech) |
| 2023 | TF-GridNet | 23.4 dB | Complex T-F LSTM (blind) |
| 2023 | AudioSep | N/A | Text query conditioning |
| 2024 | SPMamba / TIGER | ~22–23 dB | State-space model + efficient T-F (streaming) |
| 2024 | SpeakerBeam-SS | ~15 dB | SSM + speaker conditioning (real-time) |
| 2025 | GenTSE | ~18 dB | Generative LM (decoder-only) for TSE |

**Critical insight:** The highest-quality blind separation models (23+ dB) do not have speaker conditioning. The highest-quality speaker-conditioned models (SpEx+, ~17 dB) lag ~6 dB behind. Our goal is to close this gap while adding: TTS enrollment support, multilingual capability, and music robustness.

### 2.2 Gap Analysis Summary

Three critical gaps exist in the literature:

1. **TTS-as-enrollment design gap:** All existing TSE systems assume clean real enrollment audio. No system is designed and trained for synthesized enrollment. One paper (INTERSPEECH 2023) showed < 0.5 dB SI-SDR gap with TTS enrollment under ideal conditions, but no systematic study or purpose-built system exists.

2. **Multilingual TSE gap:** No published TSE work on Hindi or code-switched Hindi-English audio. The closest work is multilingual speaker verification (Kathbath, IndicSUPERB) — not extraction.

3. **Music-aware speaker conditioning gap:** Music source separation (Spleeter, DEMUCS) and speaker extraction (SpEx+, VoiceFilter) exist as separate fields. No model combines them.

### 2.3 Closest Existing Work

| Paper | Overlap | Gap |
|---|---|---|
| SpEx+ (2021) | Speaker conditioning, multi-scale encoder | No TTS support, English only, no music |
| VoiceFilter (2019) | Speaker conditioning, d-vector | No TTS, Google-internal, English |
| AudioSep (2023) | Text conditioning for audio | Generic text, not voice-specific |
| BSRNN (2022) | Music+speech handling | No speaker conditioning |
| RVAE-EM (2022) | Iterative refinement | No neural TSE, no TTS |

No single paper covers more than one of our three novel contributions simultaneously.

---

## SECTION 3: PROPOSED SYSTEM

### 3.1 Architecture (TTSE-Net)

TTSE-Net is a 34M parameter neural network for target speaker extraction with the following components:

```
Input: x(t) = mixed audio, ŝ(t) = TTS-synthesized reference

1. MULTI-SCALE ENCODER: Convert x(t) to features at 3 temporal scales
   (2.5ms, 10ms, 20ms windows) → concatenated 512-dim representation

2. SPEAKER ENCODER: Extract e_spk ∈ R^{192} from ŝ(t) via ECAPA-TDNN
   (multilingual fine-tuned; synthesis-robust via paired training)

3. T-F TEMPLATE EXTRACTOR: Extract spectral shape patterns from ŝ(t)
   via STFT + 2D convolutions → T_ref for cross-attention

4. BAND-SPLIT MODULE: Split frequency axis into 12 logarithmic bands
   (enables music/speech joint modeling)

5. BAND-SPLIT CONFORMER-LSTM (6 layers per band):
   - FiLM conditioning with e_spk at each layer
   - Cross-attention with T-F template (gated by reliability)
   - Temporal BiLSTM for sequence modeling
   - Cross-band Conformer (2 layers) for harmonic coupling

6. DECODER: Complex T-F mask → ISTFT → extracted waveform ŝ_ext(t)

7. ITERATIVE REFINEMENT (optional, offline mode):
   ŝ_ext(t) → re-embed → better conditioning → 2-3 iterations
```

**Key innovations:**
- Dual conditioning (identity embedding + spectral template) vs. existing embedding-only
- Template reliability gating (adapts when TTS quality is low)
- Band-split backbone for music robustness
- Multilingual speaker encoder fine-tuned on synthesis artifacts

### 3.2 Streaming Variant (TTSE-Net-S)

For production streaming (< 200ms latency):
- Unidirectional LSTM (replace BiLSTM)
- Chunk-based processing (200ms chunks, 50ms overlap)
- Linear attention for cross-band modeling
- Pre-computed speaker embedding (no per-chunk re-computation)
- RTF < 0.3 on CPU, < 0.02 on GPU

### 3.3 Training Recipe

**4-stage curriculum:**

| Stage | Data | Duration | Key Focus |
|---|---|---|---|
| 1 | English clean, 2-spk, real enrollment | 200k steps | Basic speaker conditioning |
| 2 | English noisy+reverb, 2-3 spk, 30% TTS | 150k steps | Noise robustness + TTS enrollment |
| 3 | Multilingual + music (40%), 60% TTS | 150k steps | Hindi + music handling |
| 4 | All conditions, hard SNR, 70% TTS | 100k steps | Final generalization |

**Loss function:** SI-SDR (primary) + multi-resolution STFT (spectral quality) + speaker similarity (identity preservation) + ASR CTC auxiliary (intelligibility, Stage 3+)

**Total compute:** ~864 A100 GPU-hours (~$1,728 on cloud at standard rates)

---

## SECTION 4: EVALUATION PLAN

### 4.1 Test Benchmark (5,800 samples)

Three language tracks:
- **English (2,900 samples):** LibriSpeech, VCTK, VoxCeleb — 8 conditions
- **Hindi (1,950 samples):** Kathbath, Shrutilipi, MLS Hindi — 7 conditions
- **Code-switched (950 samples):** MUCS 2021 — 4 conditions

Three difficulty tiers:
- **Easy:** High SNR (0–10 dB), clean, 2-speaker
- **Medium:** Moderate SNR (-5–5 dB), noise or music, 2-3 speaker
- **Hard:** Low SNR (-10–0 dB), music+noise+reverb, 3+ speaker

Four mixture types per language: clean multi-speaker, noisy, music background, all combined.

### 4.2 Metrics

| Category | Metric | Tool |
|---|---|---|
| Signal quality | SI-SDR, SDRi | fast_bss_eval |
| Perceptual quality | PESQ, ESTOI | pesq, pystoi packages |
| Intelligibility | WER/CER | Whisper large-v3 |
| Perceptual MOS | DNSMOS P.835 | Microsoft ONNX |
| Speaker fidelity | Cosine sim (ECAPA-TDNN) | SpeechBrain |

### 4.3 Ablation Study (5 Models)

Before developing TTSE-Net, run ablations to validate hypotheses:

| Model | Role | Key Question |
|---|---|---|
| SpEx+ (real enrollment) | TSE baseline | Best possible with real enrollment |
| SpEx+ (TTS enrollment) | Our deployment case | Gap from real to TTS enrollment |
| SepFormer | Blind upper bound | How much does speaker conditioning add? |
| TF-GridNet | Architecture quality ceiling | Can we match ~23 dB with conditioning? |
| BSRNN | Music robustness | Band-split advantage on music? |
| AudioSep | Text conditioning alternative | Generic text vs. voice conditioning |

**Key hypotheses:**
- H1: TTS enrollment gap < 2 dB vs. real on clean English
- H2: All models degrade > 5 dB on Hindi (confirming multilingual gap)
- H3: BSRNN > 3 dB gain over TCN on music conditions
- H4: Blind SepFormer beats SpEx+ on Hindi (language-mismatch problem)

---

## SECTION 5: TIMELINE

### Phase 0: Setup (Week 1-2)
- [ ] Git repo initialization, CI pipeline
- [ ] Data download and validation (LibriSpeech, Kathbath, Shrutilipi, MUSAN, MusDB18)
- [ ] Mixture generation code (pyroomacoustics + scaper)
- [ ] TTS pipeline setup (XTTS-v2, YourTTS, OpenVoice)

### Phase 1: Test Set Construction (Week 3-4)
- [ ] Generate all 5,800 test mixtures with `seed=42`
- [ ] Pre-synthesize TTS enrollments for all 3 TTS systems
- [ ] Validate metadata, compute reference metrics (oracle SI-SDR)

### Phase 2: Ablation Study (Week 5-9)
- [ ] Install and validate SpEx+, SepFormer, TF-GridNet, BSRNN, AudioSep
- [ ] Run all 5 models on all test conditions
- [ ] Analyze results, write ablation report
- [ ] Validate/refute H1-H5 hypotheses

### Phase 3: Architecture Development (Week 10-14)
- [ ] Implement TTSE-Net components (speaker encoder, band-split, conformer-LSTM)
- [ ] Unit tests for all modules
- [ ] Integration test on small-scale data

### Phase 4: Training (Week 15-25)
- [ ] Speaker encoder pre-training/fine-tuning (2 days)
- [ ] TTS enrollment pre-synthesis for training corpus (2 days)
- [ ] Stage 1 training (2 days)
- [ ] Stage 2 training (1.5 days)
- [ ] Stage 3 training (2 days)
- [ ] Stage 4 training + hyperparameter tuning (1.5 days + iteration)
- [ ] Streaming variant training

### Phase 5: Evaluation & Paper (Week 26-30)
- [ ] Full evaluation on 5,800-sample benchmark
- [ ] Ablation tables (architecture components, TTS systems, # iterations)
- [ ] Write paper (target: INTERSPEECH 2026 or ICASSP 2027)
- [ ] Public benchmark release

### Milestones
| Milestone | Target Date | Success Criterion |
|---|---|---|
| Test set v1.0 | Week 4 | All 5,800 mixtures generated and validated |
| Ablation report | Week 9 | H1-H5 tested; architecture decisions finalized |
| TTSE-Net Stage 1 training | Week 17 | Val EN-C1 SI-SDRi > 14 dB |
| TTSE-Net full training | Week 25 | Val HI-C1 SI-SDRi > 11 dB, EN-M1 > 10 dB |
| Final evaluation | Week 28 | Full benchmark results; competitive with SpEx+ |
| Paper submitted | Week 30 | INTERSPEECH 2026 deadline |

---

## SECTION 6: OPEN QUESTIONS & RISKS

### Technical Risks

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| TTS enrollment gap > 2 dB | High | Medium | Additional synthesis-robustness fine-tuning; multiple TTS systems |
| Hindi speaker encoder weak | High | High (near-certain) | Fine-tune ECAPA on IndicSUPERB; consider mHuBERT-based encoder |
| Music backgrounds cause T-F template confusion | Medium | Medium | Template reliability gate; BSRNN bands |
| SI-SDR with conditioning < blind SepFormer | High | Low-Medium | If this happens: adopt SepFormer backbone, add post-hoc conditioning |
| Shrutilipi data quality issues | Medium | Medium | Manual quality filtering; subset to high-quality recordings |
| Code-switched model confusion | Medium | High | Explicit CS data from Week 1; language ID pre-processor |

### Data Risks

| Risk | Severity | Mitigation |
|---|---|---|
| MusDB18 CC-BY-NC-SA limits commercial use | High | Use MUSAN music (CC-BY) for commercial product; MusDB for research |
| Shrutilipi license ambiguity | Medium | Contact AI4Bharat; use CC-BY confirmed Kathbath/MLS as backup |
| VoxCeleb downloads restricted | Low | Use LibriSpeech + VCTK; VoxCeleb adds diversity |

### Research Risks

| Risk | Mitigation |
|---|---|
| Someone publishes similar work first | Speed up ablation phase; pre-print on ArXiv after test set release |
| T-F template doesn't add value over embedding | Architecture is modular; remove template branch if ablation shows no gain |
| Iterative refinement doesn't converge well | Limit to 1 iteration; focus on single-pass quality |

### Open Questions (To Be Answered by Ablation)

1. **Is dual conditioning (embedding + template) actually better than embedding-only?** This is the core hypothesis of the T-F template branch. If ablation shows < 0.5 dB gain, simplify to embedding-only.

2. **Which TTS system produces the best enrollment quality for extraction?** Compare XTTS-v2 vs. YourTTS vs. OpenVoice systematically. The winner becomes the recommended TTS for production.

3. **How many refinement iterations are needed?** Validate convergence curve: 1, 2, 3 iterations vs. quality.

4. **Can one model handle English + Hindi + code-switched equally well, or do we need separate models?** Curriculum training hypothesis: one model with multilingual training should converge. If not, Hindi-specific fine-tune.

5. **What is the minimum enrollment audio quality for viable extraction?** Test: real clean enrollment vs. noisy enrollment vs. TTS enrollment from low-quality voice sample. Informs product spec for minimum input quality.

---

## APPENDIX A: DIRECTORY STRUCTURE

```
speech_separation_research/
├── FINAL_RESEARCH_PLAN.md          (this document)
├── fetch_log.txt                    (web request log)
├── research/
│   ├── literature_survey.md         (Task 1: papers, repos, models)
│   ├── sota_audit.md                (Task 2: model comparison table + ranking)
│   ├── gap_analysis.md              (Task 3: novelty assessment)
│   ├── test_set_design.md           (Task 4: benchmark specification)
│   └── ablation_plan.md             (Task 5: evaluation plan for existing models)
└── model/
    ├── architecture_design.md       (Task 6: TTSE-Net design)
    └── training_plan.md             (Task 7: full training recipe)
```

## APPENDIX B: KEY REFERENCES

1. Wang, Q. et al. "VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking." INTERSPEECH 2019. arXiv:1810.04826
2. Ge, M. et al. "SpEx+: A Complete Speech Extraction Neural Network." arXiv:2004.14948
3. Subakan, C. et al. "Attention is All You Need in Speech Separation." ICASSP 2021. arXiv:2010.13154
4. Wang, Z.Q. et al. "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation." arXiv:2209.03952
5. Luo, Y. et al. "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation." IEEE TASLP 2019. arXiv:1809.07454
6. Liu, Y. et al. "BSRNN: Band-Split RNN for Monaural Music Source Separation." arXiv:2209.15174
7. Liu, H. et al. "AudioSep: Separating Anything You Describe." NeurIPS 2023. arXiv:2308.05037
8. Richter, J. et al. "SGMSE+: Speech Enhancement and Dereverberation with Diffusion-Based Generative Models." arXiv:2208.05830
9. Žmolíková, K. et al. "SpeakerBeam: Speaker Aware Neural Network for Target Speaker Extraction." IEEE JSTSP 2019.
10. Babu, A. et al. "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale." arXiv:2111.09296

---

*This document is a complete research plan ready for engineering handoff. All source files are in `./speech_separation_research/`. Begin with Phase 0 (Setup) as described in Section 5.*
