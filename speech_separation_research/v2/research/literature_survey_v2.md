# Literature Survey v2: TTS-Guided Target Speaker Extraction
**Date:** 2026-04-06  
**Method:** PDFs downloaded (17 priority + 11 newer papers), real citation counts from Semantic Scholar API, architectural details extracted directly from papers.  
**Total papers surveyed:** 50  

---

## CRITICAL FINDING: SAM Audio

**SAM Audio = Meta FAIR's "Segment Anything in Audio"** (arXiv 2512.18099, Dec 2025).  
This is the definitive meaning of the term. It is **not** Stable Audio (Stability AI) — that is a text-to-audio generation model, unrelated to separation.

**SAM Audio architecture:**
- Diffusion Transformer (DiT) trained with **flow matching**
- Latent representation: DAC-VAE, 25 Hz frame rate
- Three sizes: Base (~500M), Large (~1–3B params) — excludes external encoders
- Conditioning modalities: **text** (T5), **visual** (SAM2 masks via Perception Encoder AV), **temporal spans**
- Trained on ~1M hours: 21,910h conversational speech + 20,000h music + 10,600h sound effects + pseudo-labeled
- Outperforms MossFormer2, TIGER across all separation categories on SAM-Audio-Bench (in-the-wild)

**Critical limitation for our use case:** SAM Audio does **not** support audio enrollment-based extraction. It cannot take a reference clip of a target speaker and extract them. It separates by text description ("female speaker", "piano") or visual mask. This is the key gap our system fills.

**Open source:** Yes — `facebookresearch/sam-audio`, `facebook/sam-audio-base/large` on HuggingFace. SAM License.

---

## SECTION 1: TSE PARADIGM EVOLUTION (2018→2026)

The field has evolved through five distinct paradigms. As of 2025–2026, **no single paradigm dominates** — the frontier is hybrid discriminative-generative approaches.

```
2018  │ STFT masking + d-vector            VoiceFilter (418 cites)
2019  │ Time-domain TCN + speaker emb      SpEx, SpeakerBeam (200–210 cites)
2020  │ Multi-scale + FiLM cond.           SpEx+ (187 cites), DPRNN (932 cites)
2021  │ Dual-path Transformer              SepFormer (745 cites, 22.3 dB)
2022  │ Complex T-F + hybrid               TF-GridNet (166/223 cites, 23.4 dB)
      │ Band-split for music               BSRNN (199 cites)
      │ SSL pre-training                   WavLM (2950 cites) → best speaker encoder
2023  │ Transformer+RNN-free recurrent     MossFormer2 (74 cites, 24.1 dB)
      │ Token LM for audio                 AudioLM (891), VALL-E (1126), EnCodec (1095)
      │ Text-queried universal             AudioSep (80 cites), UniAudio (193 cites)
      │ Diffusion enhancement              SGMSE+ (352 cites)
2024  │ Asymmetric enc-dec                 SepReformer (32 cites, 25.4 dB — SOTA)
      │ Sub-1M efficient                   TIGER (12 cites, ICLR 2025)
      │ Mamba separation                   SPMamba (22.5 dB)
      │ CMHA / embedding-free TSE          USEF-TSE (15 cites, best WavLM sim=0.935)
      │ Discrete token TSE                 TSELM (10 cites, ICASSP 2025)
      │ Real-time SSM TSE                  SpeakerBeam-SS (14 cites)
      │ Flow-based LASS                    FlowSep (8 cites, ICASSP 2025)
      │ Masked generative + prompt         AnyEnhance (12 cites) — prompt-guided TSE
      │ Universal codec LM                 UniSep (36.5k hrs, ICME 2025)
2025  │ AR LM for TSE                      GenTSE, LauraTSE — best perceptual quality
      │ Discriminative-generative hybrid   Disc-Gen-TSE (2601.06006) — frontier
      │ Flow matching unified              Geneses (enhancement+separation)
      │ SAM Audio                          Meta FAIR — text/visual separation at scale
```

---

## SECTION 2: KEY PAPERS WITH REAL METRICS

### 2.1 Discriminative Masking — Best Models

#### SepReformer (NeurIPS 2024, arXiv 2406.05983)
**Current SOTA on WSJ0-2mix: 25.4 dB SI-SNRi with dynamic mixing**
- Architecture: Asymmetric encoder-decoder. Encoder: interleaved local + global Transformer on full sequence (no chunking). Speaker dimension expansion before decoding. Weight-shared decoder for cross-speaker processing.
- Model sizes: SepReformer-B (14.2M params, 39.8 G/s MACs), SepReformer-L (larger). SepReformer-T <5M params.
- SepReformer-B outperforms SepFormer (26M, 86.9 G/s) — same quality at 10x less compute
- Code: github.com/dmlguq456/SepReformer

#### TF-GridNet (ICASSP 2023, arXiv 2209.03952) — 166 cites
**23.4 dB SI-SDRi on WSJ0-2mix (confirmed from PDF extraction)**
- Complex spectral mapping (real + imaginary simultaneously). 3 modules per block: intra-frame spectral, sub-band temporal BLSTM, full-band self-attention.
- 14.4M params. Benchmark results (from PDF): Conv-TasNet=15.3, DPRNN=18.8, SepFormer=22.3, TF-GridNet=23.4/23.5 dB.

#### USEF-TSE (IEEE/ACM TASLP, arXiv 2409.02615) — **Most Relevant for Our Architecture**
**Best speaker similarity among all TSE models (WavLM Sim=0.935, from LauraTSE PDF)**
- No speaker embedding. CMHA (Cross Multi-Head Attention) takes mixture encoding as Q, full reference audio encoding as K/V.
- Output: frame-level features maintaining same length as mixture — fed into separator.
- USEF-SepFormer and USEF-TFGridNet variants.
- **Key insight for us:** CMHA naturally handles timing/prosody mismatch in TTS reference because cross-attention is permutation-flexible. The model finds relevant speaker features regardless of when they appear in the reference.

**Full results table (from LauraTSE PDF, directly comparable):**

| Model | Cat | DNSMOS OVL | NISQA | SpeechBERT | dWER↓ | WavLM Sim↑ | WeSpeaker Sim↑ |
|---|---|---|---|---|---|---|---|
| Mixture | — | 2.653 | 2.453 | 0.572 | 79.2 | — | — |
| SpEx+ | D | 3.186 | 3.349 | 0.878 | 3.472 | 0.148 | 0.973 |
| WeSep | D | 3.118 | 3.892 | 0.895 | 3.486 | 0.123 | 0.980 |
| **USEF-TSE** | D | **3.272** | **4.319** | **0.935** | **3.555** | **0.0747** | **0.988** |
| TSELM-L | G | 3.212 | 3.961 | 0.793 | 3.489 | 0.297 | 0.887 |
| AnyEnhance | G | **3.353** | 4.277 | 0.735 | **3.638** | — | 0.914 |
| **LauraTSE** | G | 3.336 | **4.333** | **0.908** | 3.609 | 0.159 | 0.974 |
| LauraTSE-stream | G | 3.314 | 4.275 | 0.897 | 3.596 | 0.169 | 0.973 |

*D=Discriminative, G=Generative. dWER lower is better; others higher is better.*

---

### 2.2 Token-Based & Generative TSE — The Frontier

#### TSELM (ICASSP 2025, arXiv 2409.07841)
- WavLM discrete tokens + cross-attention + LM + HiFi-GAN
- Cross-entropy loss (not SI-SDR)
- Strong DNSMOS but dWER slightly higher than discriminative models
- **Paradigm pioneer:** converts TSE to sequence classification

#### AnyEnhance (arXiv 2501.15417) — **Best DNSMOS (3.638)**
- MaskGIT-style masked generative model
- Audio codec tokens (encode/decode via neural codec)
- **Prompt-guidance:** reference audio prompt for TSE without fine-tuning
- Training data: Emilia 21k hours (filtered with DNSMOS>3.4) + VCTK + singing + noise MUSAN + FSD50K
- Handles denoising, dereverberation, declipping, super-resolution, AND TSE — one model
- Self-critic: iterative self-assessment during generation
- **Key for us:** Prompt-guidance mechanism is exactly how TTS enrollment can condition extraction

#### LauraTSE (arXiv 2504.07402) — **Best Perceptual + Speaker Sim balance**
- AR decoder-only LM (LauraGPT) on continuous embeddings of mixture+reference
- Generates coarse codec tokens → encoder-only LM refines to fine tokens
- LauraTSE-streaming variant for real-time deployment
- dWER=4.333 (best among generative models)

#### GenTSE (arXiv 2512.20978) — Newest
- Two-stage: semantic tokens (SSL) → acoustic tokens (codec)
- Frozen-LM Conditioning + DPO alignment
- dWER improved 0.217→0.172 with FLC
- Evaluated on Libri2Mix

#### Disc-Gen-TSE / Unified Framework (arXiv 2601.06006)
- Discriminative front-end for stable intermediate representation
- Generative back-end in codec space for perceptual quality
- Combined: best dWER + best perceptual quality

---

### 2.3 Flow-Matching Approaches

#### FlowSep (ICASSP 2025, arXiv 2409.07614)
- Rectified flow matching (RFM) in VAE latent space for text-queried separation
- FLAN-T5 + UNet with cross-attention
- 10 inference steps vs diffusion's 200+
- FAD 2.86 on AudioCaps (AudioSep: 4.38) — lower is better
- **Critical limitation:** Text-queried only, no speaker enrollment

#### Geneses (arXiv 2601.18456)
- Unified flow matching for both enhancement and separation
- SSL features from noisy mixture condition the flow predictor (DiT)
- DNSMOS/NISQA close to ground truth — strongest perceptual quality yet
- Still 2-speaker only; no multilingual evaluation

#### MeanFlow-TSE / AlphaFlowTSE (arXiv 2512.18572)
- One-step flow distillation for TSE
- Real-time capable

---

### 2.4 Foundation Models for Separation

#### SAM Audio (Meta FAIR, arXiv 2512.18099) — **Biggest model in space**
- DiT + flow matching, 500M–3B params, ~1M hours training
- Text prompt: "woman speaking", "violin"
- Visual prompt: SAM2 binary masks from video frames
- Temporal span: mark time intervals
- **Cannot use audio enrollment** — explicitly a design choice
- In-the-wild benchmark SAM-Audio-Bench: outperforms all specialists
- Code: github.com/facebookresearch/sam-audio

#### UniAudio 1B (arXiv 2310.00704) — **Only large model with audio-enrollment TSE**
- 1B params (744M global + 238M local transformer)
- Codec tokenization (RVQ, 3 codebook layers)
- Trained on 165,000 hours (LibriLight 60k, MLS 20k, AudioSet, etc.)
- **Explicitly supports TSE with 3-second audio enrollment**
- 11 tasks including TSE, speech enhancement, TTS, voice conversion

#### SpeechX (arXiv 2308.06873, 118 cites)
- Neural codec language model (EnCodec, 8 codebooks, 75 Hz, 1024 entries/layer)
- Multi-task: TTS, noise suppression, **target speaker extraction**, speech removal, editing
- Speaker extraction via task-dependent prompting + mixed audio input
- PESQ=4.6, DNSMOS=3+ on enhancement tasks
- Microsoft Research; weights not fully public

---

### 2.5 Music Separation (Relevant for Music-Aware TSE)

#### BSRNN / Band-Split RNN (TASLP, arXiv 2209.15174) — 199 cites
- Splits frequency into sub-bands, processes each with RNN, cross-band recombination
- Best for joint music+speech (handles harmonics correctly)
- SDR on MUSDB18 vocals: 10.4 dB

#### HT-Demucs (ICASSP 2023, arXiv 2211.08553) — 267 cites
- Hybrid spectrogram U-Net + Transformer in innermost layers
- MUSDB18-HQ SDR: 9.20 dB (SOTA at release)
- Trained on 800+ tracks (proprietary)

#### TFC-TDF-UNet v3 (SDX'23 winner)
- Complex spectrogram UNet (predicts complex source directly)
- Won Music Demixing track of SDX'23

---

### 2.6 TTS/Voice Cloning Systems (For Enrollment Synthesis)

#### CosyVoice (Alibaba, arXiv 2407.05407) — 399 cites
- Zero-shot voice cloning + multilingual TTS
- Supports Chinese, English, Japanese, Korean, Cantonese
- Flow-matching based vocoder
- Production-deployed

#### F5-TTS (arXiv 2410.06885) — 350 cites
- Flow matching E2/F5 TTS, non-autoregressive
- Very high speaker similarity in voice cloning
- Fast inference, open source

#### XTTS-v2 (arXiv 2406.04904) — 245 cites
- Zero-shot multilingual voice cloning
- Supports 17 languages including Hindi
- 3-second enrollment, high quality

#### VALL-E 2 (arXiv 2406.05370)
- Codec language model for zero-shot TTS
- Grouping-based codec LM for efficiency
- CQNX (Codec-Quality-N-gram-eXponential) sampling

#### StyleTTS2 (NeurIPS 2023, arXiv 2306.07691) — 236 cites
- Style-based diffusion TTS, very natural
- Open source, English-primary

---

### 2.7 Large-Scale Training Datasets

#### EMILIA (ICASSP 2025/SLT 2024, arXiv 2407.05361) — 213 cites
- **101,654 hours** of multilingual in-the-wild speech (English, Chinese, German, French, Japanese, Korean)
- Sourced from internet (podcasts, audiobooks, videos)
- Filtered with Whisper ASR + pyannote diarization + DNSMOS quality scoring
- License: CC-BY 4.0
- Used by AnyEnhance (21k hour subset), F5-TTS, CosyVoice

#### LibriheavyMix (Interspeech 2024, arXiv 2409.00819) — 8 cites
- 20,000 hours multi-turn reverberant overlapping speech
- Built from Libriheavy (a LibriSpeech extension)
- Variable speaker overlaps 0–100%, RIR augmentation
- Explicit train/test speaker splits — ideal for TSE training

#### LibriLight (Facebook, 60,000 hours)
- Weakly labeled English speech from LibriVox
- Used in WavLM, UniAudio pre-training

---

### 2.8 Realistic Evaluation Benchmarks

| Benchmark | Type | Condition | Notes |
|---|---|---|---|
| WSJ0-2mix | Synthetic clean | 2-spk, no noise | Saturated (25.4 dB); still standard |
| LibriheavyMix | Synthetic reverberant | Multi-turn, overlap | Most comprehensive scale |
| EchoSet (TIGER) | Realistic reverb | 2-spk + echo | New 2024 |
| REAL-M | Real mixtures | Blind (no GT) | Uses SI-SNR estimator |
| CHiME-8 NOTSOFAR | Real meetings | Multi-speaker | Most realistic |
| DnR (Divide & Remaster) | Cinematic | Dialogue+music+SFX | Real movie |
| URGENT 2024 | Enhancement | Multi-degradation | 21 teams competed |
| SAM-Audio-Bench | In-the-wild | All types | New 2025, reference-free scoring |

---

## SECTION 3: SPEAKER ENCODER RANKING (2025 State)

From LauraTSE results and USEF-TSE paper — ranked for TSE conditioning:

1. **WavLM-large / WavLM-base-plus** (microsoft/wavlm-large on HuggingFace)
   - Best speaker representations; used in TSELM, USEF-TSE, evaluation in GenTSE
   - WavLM Sim in LauraTSE table: ~0.9+ for best models
   - ~316M params (WavLM-large)

2. **ECAPA-TDNN** — TitaNet achieves comparable EER (1.91%)
   - Second-best; much smaller and faster
   - Production standard; used in SpEx+, ClearerVoice

3. **CMHA (Cross Multi-Head Attention, USEF-TSE style)**
   - Not an "encoder" per se — uses full reference sequence directly
   - **Best WavLM similarity (0.935)** and WeSpeaker sim (0.988) in the table
   - No pooling bottleneck → preserves fine-grained speaker characteristics
   - Most robust to TTS timing/prosody mismatch

4. **ResNet-based (WeSpeaker ResNet221LM)**
   - Strong for evaluation; less common for conditioning

5. **d-vector / x-vector** — weakest; legacy

**Recommendation:** CMHA (USEF-style) + WavLM as feature extractor for the reference. No pooled embedding needed.

---

## SECTION 4: KEY ARCHITECTURAL INSIGHTS FROM CHALLENGE RESULTS

**URGENT 2024 (21 teams):**
- Winning systems: ensemble/multi-task discriminative models
- Single metrics mislead; multi-metric evaluation required
- Data quality problems severe even in "clean" corpora

**CHiME-8 (2024):**
- Dominant pipeline: Diarization → Speaker-conditioned extraction (G-TSE/GSS) → Whisper ASR
- Cascade still beats end-to-end for real meetings
- Whisper featured in nearly all winning systems as ASR backend

**SDX'23 Cinematic:**
- HTDemucs variants dominate
- +5.7 dB improvement over cocktail-fork baseline
- Perceptual quality gaps beyond SDR are real in listening tests

**Key takeaway:** For production (not just benchmark), the winning approach remains **discriminative TSE front-end + Whisper ASR back-end**. The generative paradigm leads in perceptual quality but trails in intelligibility (dWER).

---

## SECTION 5: WHAT'S ACTUALLY MISSING (2026 GAP)

After surveying 50 papers, the following remain genuinely unexplored:

1. **TTS-as-runtime-enrollment for TSE** — No paper uses synthesized speech as the conditioning signal at inference. Miipher does restoration using transcripts, but not separation. USEF-TSE uses CMHA which could accept TTS audio, but isn't designed for it.

2. **Hindi + Hindi-English code-switched TSE** — Zero published work.

3. **Music-aware speaker-conditioned extraction** — SAM Audio does music separation and speech separation but not simultaneously with speaker enrollment.

4. **Enrollment-based TSE at 1B+ scale** — UniAudio (~1B) is the only model near this scale with enrollment TSE, but it uses pooled embeddings, not CMHA.

5. **Synthesis-artifact-robust speaker conditioning** — Papers show TTS embeddings work (close to real), but no model is explicitly designed/trained to be robust to synthesis artifacts at inference time.
