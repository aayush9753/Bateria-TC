I now have comprehensive coverage of all the requested topics. Let me compile the full research report.

---

# Deep Research Report: Target Speaker Extraction, Speech Separation & Generative Audio Separation (2024–2025)

---

## PART I: MUST-FIND PAPERS — FULL SUMMARIES

---

### 1. arXiv 2301.13341 — "Neural Target Speech Extraction: An Overview"

**Title:** Neural Target Speech Extraction: An Overview
**Authors:** Katerina Zmolikova, Marc Delcroix, Tsubasa Ochiai, Keisuke Kinoshita, Jan Cernocky, Dong Yu
**Year:** 2023 (IEEE Signal Processing Magazine, accepted Jan 2023)
**ArXiv:** [2301.13341](https://arxiv.org/abs/2301.13341)

**Full Taxonomy of TSE Clue Types:**

The paper organizes TSE into a three-axis taxonomy:

**Axis 1: Clue Modality**
- *Enrollment speech / d-vector*: Pre-recorded utterance from target speaker, most common lab setup
- *Spatial / microphone array*: Direction-of-arrival, beamforming-based clues
- *Visual*: Lip movement, face identity (AV-TSE)
- *Brain signals (EEG)*: Neuro-steered extraction, the most experimental branch
- *Combined / multi-cue*: Fusion of multiple modalities

**Axis 2: Network Architecture**
- Conditioning-in-encoder (e.g., SpeakerBeam): Speaker embedding injected at the front
- Conditioning-in-separator (e.g., SpEx, SpEx+): Speaker embedding interleaved at each processing block
- Conditioning-in-decoder: Speaker cue applied during waveform reconstruction
- Attention-based fusion: Cross-attention between mixture and speaker embedding

**Axis 3: Speaker Representation**
- Fixed pre-trained speaker encoder (x-vector, d-vector, ECAPA-TDNN)
- Jointly-trained speaker encoder
- End-to-end (no explicit speaker encoder, direct cross-attention to enrollment waveform)

**Key cited models:** SpEx, SpEx+, SpeakerBeam, TD-SpeakerBeam, VoiceFilter, DPCTNET-TSE, TENet

**Limitations:** Does not cover generative TSE (post-2023), flow-matching, or LLM-based approaches.

---

### 2. Flow-Matching / Consistency-Model Based Speech Separation (2024–2025)

Several distinct lines:

#### 2a. FlowSep (ICASSP 2025)
**Title:** FlowSep: Language-Queried Sound Separation with Rectified Flow Matching
**ArXiv:** [2409.07614](https://arxiv.org/abs/2409.07614)
**Key method:** Trains a UNet with cross-attention in VAE latent space (from AudioLDM's VAE) using Rectified Flow Matching (RFM). Text queries encoded by FLAN-T5. Mixture audio latent is channel-concatenated with noise as conditioning. Uses BigVGAN vocoder. Avoids the "spectral holes" artifacts of mask-based systems.
**Training data:** 1,680 hours (AudioCaps 49k clips, VGGSound 200k clips, WavCaps 400k clips)
**Metrics:** FAD 2.86 on AudioCaps (vs. AudioSep 4.38); Relevance 4.08–4.11 / OVL 3.72–3.98 (vs. AudioSep 3.66–3.93/2.69–3.53). Does not report SDR — authors argue SDR is ill-suited for generative separation.
**Inference speed:** 10 steps vs. diffusion's 200+
**Code:** [audio-agi.github.io/FlowSep_demo](https://audio-agi.github.io/FlowSep_demo/)
**Limitations:** Language-queried (not enrollment-based TSE); metrics are perceptual not SI-SDR; model size undisclosed.

#### 2b. FlowTSE (May 2025)
**ArXiv:** [2505.14465](https://arxiv.org/abs/2505.14465)
Conditional flow matching applied directly to speaker-enrollment TSE. Operates on mel-spectrograms. Simpler pipeline than diffusion-based predecessors.

#### 2c. MeanFlow-TSE (Dec 2025)
**ArXiv:** [2512.18572](https://arxiv.org/abs/2512.18572)
One-step generative TSE using mean-flow objectives. Eliminates iterative refinement. Achieves competitive performance in single forward pass. SI-SDR 12.85 dB on Libri2Mix Noisy (PESQ 2.21). Outperforms diffusion/flow multi-step TSE models on perceptual metrics.

#### 2d. AD-FlowTSE / Adaptive Deterministic Flow Matching TSE (Oct 2025)
**ArXiv:** [2510.16995](https://arxiv.org/abs/2510.16995)
Defines flow between background and source conditioned on mixing ratio. Can run in a single step while adapting step size dynamically.

#### 2e. Geneses — Unified Generative SE+SS (Jan 2026)
**ArXiv:** [2601.18456](https://arxiv.org/abs/2601.18456)
Latent flow matching in VAE space conditioned on SSL features (from noisy mixture). Multi-modal diffusion Transformer. Handles both enhancement (denoising) and separation. Significantly outperforms mask-based baselines on DNSMOS, NISQA, UTMOSv2 metrics, approaching Ground Truth perceptual scores. Evaluated on LibriTTS-R two-speaker mixtures.

---

### 3. ClearerVoice-Studio (Alibaba Speech Lab, 2024)

**Title:** ClearerVoice-Studio: Bridging Advanced Speech Processing Research and Practical Deployment
**ArXiv:** [2506.19398](https://arxiv.org/abs/2506.19398) (also GitHub)
**Code:** [github.com/modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
**HuggingFace demo:** [alibabasglab/ClearVoice](https://huggingface.co/spaces/alibabasglab/ClearVoice)

**Technical Stack — Core Models:**

| Task | Model | Notes |
|---|---|---|
| Speech Denoising (16kHz) | FRCRN | 2022 DNS Challenge 2nd place |
| Speech Enhancement (48kHz) | MossFormer2-48k | Sub-band + FSMN hybrid |
| Speech Separation (2-spk) | MossFormer2 | SOTA on WSJ0-2mix, Libri2Mix |
| Target Speaker Extraction | USEF-TSE variants | Embedding-free cross-attention |
| Speech Super-Resolution | MossFormer2-SR-48K | Bandwidth extension to 48kHz |
| AV-TSE | VisualVoice/AV-USEF | Face + gesture conditioning |
| Neuro-TSE | EEG-USEF | Brain-signal steered extraction |

**MossFormer2 Architecture (arXiv 2312.11825, ICASSP 2024):**
- Gated single-head Transformer with convolution-augmented joint self-attentions (from original MossFormer)
- **New in MossFormer2:** Adds an RNN-free recurrent module based on Feedforward Sequential Memory Network (FSMN) using gated convolutional units (GCU) + dense connections + dilated structure
- Handles both long-range coarse-scale dependencies (Transformer) and fine-scale recurrent patterns (FSMN)
- Achieves +1.3 dB SI-SNR over MossFormer on WSJ0-2/3mix
- State-of-the-art on WSJ0-2/3mix, Libri2Mix, WHAM!, WHAMR!
- Deployment: FRCRN used 3M+ times on ModelScope; MossFormer separator used 2.5M+ times

**Deployment scale:** 3M+ FRCRN uses, 2.5M+ MossFormer uses on ModelScope
**Limitations:** Primarily deterministic/discriminative; no generative backbone; no multilingual explicit support.

---

### 4. Miipher — Google Speech Restoration (2023)

**Title:** Miipher: A Robust Speech Restoration Model Integrating Self-Supervised Speech and Text Representations
**ArXiv:** [2303.01664](https://arxiv.org/abs/2303.01664)
**Venue:** WASPAA 2023
**Google page:** [research.google/pubs/miipher](https://research.google/pubs/miipher-a-robust-speech-restoration-model-integrating-self-supervised-speech-representation-and-linguistic-features/)

**Architecture:**
- **Input features:** w2v-BERT self-supervised speech embeddings (robust to distortion)
- **Linguistic conditioning:** PnG-BERT text encoder takes transcript → linguistic conditioning vector fed to restoration network
- **Output:** WaveRNN/WaveFit vocoder for waveform synthesis
- Core innovation: conditioning on both acoustic (w2v-BERT) and linguistic (PnG-BERT) representations makes the model robust to novel degradations; language knowledge acts as a constraint to avoid hallucinating wrong words
- Primary application: cleaning web-scraped speech to studio quality for TTS training data

**Miipher-2 (2025, arXiv 2505.04457):**
- Replaces w2v-BERT with Google's Universal Speech Model (USM, 300+ languages)
- Adds parallel adapters predicting clean USM features from noisy input
- Uses WaveFit neural vocoder
- Real-time factor: 0.0078 — can process 1 million hours in ~3 days on 100 lite TPUs
- Designed for million-hour scale data cleaning for LLM training pipelines
- Open PyTorch implementation: [github.com/yukara-ikemiya/Open-Miipher-2](https://github.com/yukara-ikemiya/Open-Miipher-2)

**Limitations:** Designed for restoration (single speaker), not multi-speaker separation; requires ASR transcript; no open model weights for original Miipher.

---

### 5. Sound Demixing Challenge 2023 (SDX'23) — Winning Architectures

#### Music Demixing Track (MDXC)
**Paper:** arXiv [2308.06979](https://arxiv.org/abs/2308.06979) — *Transactions of ISMIR*

**Key result:** Best system improved +1.6 dB SDR over the 2021 MDX winner on MDXDB21.

**Top Architectures:**
| Rank/Track | Architecture | Key Innovation |
|---|---|---|
| Winner (MDXDB21) | TFC-TDF-UNet v3 (arXiv 2306.09382) | Complex spectrogram UNet; estimates complex STFT rather than masking; trained with label-noise augmentation |
| Runner-up | Wavelet HTDemucs (WHTDemucs) | 3-branch HTDemucs: waveform + STFT + DWT branch |
| Other notable | DWT Transformer UNet | Two-branch (waveform + DWT) with cross-transformer |

**Novel challenge structure:** Introduced two training sets with intentional errors — SDXDB23_LabelNoise (labeling errors) and SDXDB23_Bleeding (audio leakage between stems) — to test robustness to real-world training data corruption.
**Evaluation:** Objective SDR + perceptual listening test by professional producers/musicians.

#### Cinematic Demixing Track (CDX)
**Paper:** arXiv [2308.06981](https://arxiv.org/abs/2308.06981)
- **Leaderboard A winner:** Team "aim-less", SDR 4.345 dB
- **Leaderboard B winner:** JusperLee, SDR 8.181 dB
- **Dominant architecture:** HTDemucs trained on Divide and Remaster (DnR) dataset; best open-set system improved +5.7 dB over cocktail-fork baseline

---

### 6. URGENT Challenge 2024 — Universal Speech Enhancement

**Paper:** arXiv [2406.04660](https://arxiv.org/abs/2406.04660) + Lessons Learned: [2506.01611](https://arxiv.org/abs/2506.01611)
**Website:** [urgent-challenge.github.io/urgent2024](https://urgent-challenge.github.io/urgent2024/)

**Task Definition:** Single unified model must handle: denoising, dereverberation, bandwidth extension (BWE), and declipping across 8kHz/16kHz/24kHz/48kHz sampling rates. This is harder than single-task SE.

**Key facts:**
- 21 teams submitted final blind-test systems
- Strong baselines: TF-GridNet, BSRNN (Band-Split RNN) — 14 teams improved over these
- **12 metrics** used: DNSMOS, NISQA, PESQ, ESTOI, speech MOS variants, plus downstream metrics
- **Key lesson:** All top systems used multi-task training; TF-GridNet + BSRNN hybrids dominated; diffusion/generative systems underperformed on objective metrics but won on perceptual quality

**Key findings from Lessons Learned paper:**
1. Label noise even in "high-quality" corpora was pervasive
2. No single system dominated across all four distortion types
3. Hardest conditions: speech overlap + strong noise simultaneously — no existing system handled this well
4. Objective metrics (PESQ, DNSMOS) do not always correlate with human preference scores

**URGENT 2025 (Interspeech 2025):** Follow-up challenge extended; paper arXiv [2505.23212](https://arxiv.org/abs/2505.23212).

---

### 7. CHiME-8 Challenge 2024 — Multi-Condition ASR/Diarization

**Website:** [chimechallenge.org/challenges/chime8](https://www.chimechallenge.org/challenges/chime8/index)
**Paper:** ISCA Archive / arXiv

**Three Tasks:**
1. **DASR** (Task 1): Generalizable + array-agnostic distant ASR + diarization
2. **NOTSOFAR-1** (Task 2): Single- and multi-channel meeting transcription
3. **MMCSG** (Task 3): Multi-modal conversation scene graph

**NOTSOFAR Multi-channel Track Winners:**
- **1st:** USTC (Niu et al.) — *Dia-Sep-ASR* pipeline: diarization → guided source separation (GSS) → ASR. Key component: Guided Target Speaker Extraction (G-TSE) model.
- **2nd:** STCON (Mitrofanov et al.) — designed G-TSE + GSS combination
- **3rd:** NTT (Kamo et al.)

**NOTSOFAR Single-channel Track:**
- **1st:** Fano Labs — DCF-DS method

**Universal pattern:** Almost all top systems combined Whisper (as ASR backbone), a neural separation/diarization front-end, and speaker-attributed ASR pipeline. Whisper featured in "most submitted systems."

**Special prize:** BUT + HLTCOE team for most novel/efficient submission.

---

### 8. SepReformer (NeurIPS 2024)

**Title:** Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation
**ArXiv:** [2406.05983](https://arxiv.org/abs/2406.05983)
**Venue:** NeurIPS 2024
**Code:** [github.com/dmlguq456/SepReformer](https://github.com/dmlguq456/SepReformer)

**Architecture (Asymmetric Encoder-Decoder):**
- **Encoder:** Analyzes full mixture representation using both local and global Transformer blocks (handles long sequences without chunking or dual-path tricks)
- **Speaker Split Module:** Expands feature sequence to N-speaker dimension — separation happens at a structural/architectural level, not by masking
- **Decoder (weight-shared):** Shared across speakers; performs cross-speaker interaction during reconstruction; reconstructs each speaker's waveform
- Avoids the computational overhead of dual-path processing (DPRNN, DPTNet)

**Key benchmark results (SI-SNRi):**
| Dataset | Model | SI-SNRi |
|---|---|---|
| WSJ0-2mix | SepReformer-L + DM | 25.1–25.4 dB |
| WSJ0-2mix | SepReformer-B (14.2M params, 39.8 G/s) | Outperforms SepFormer (26M, 86.9 G/s) |
| WHAM! | SepReformer-T | Outperforms SepFormer with 10x lower compute |
| WHAMR! | SepReformer-T | SOTA at tiny scale |

**Model variants:** SepReformer-T (tiny), -B (base), -L (large). SepReformer-B achieves comparable or better performance than SepFormer with dramatically fewer MACs.
**Limitations:** Speaker-count must be known; focused on clean/lab benchmarks; no TSE capability inherently.

---

### 9. Whisper-Based Speaker Extraction

**Key papers:**

**"Empowering Whisper as a Joint Multi-Talker and Target-Talker Speech Recognition" (arXiv 2407.09817, 2024):**
- Freezes Whisper encoder; inserts a "Sidecar separator" into the encoder to separate mixed speaker embeddings at the feature level
- Target Talker Identifier (TTI) identifies which embedding flow belongs to the enrolled target speaker
- Enables Whisper to jointly transcribe all speakers OR extract/transcribe a single target — without retraining Whisper weights

**"Enhancing Intelligibility for Generative TSE via Joint Optimization with Target Speaker ASR" (arXiv 2501.14477, 2025):**
- Builds a generative TSE pipeline around Whisper as the ASR backbone
- Joint optimization of separation + ASR objectives
- Whisper's semantic knowledge constrains generated audio to be intelligible

**Practical pattern in CHiME-8:** Whisper used as ASR backbone in virtually all winning pipelines, paired with GSS or neural separation front-end.

---

## PART II: PARADIGM COMPARISON — BEST REPRESENTATIVE PAPERS

---

### A. Discriminative Masking: TF-GridNet & SepReformer

**Best paper for discriminative masking:** TF-GridNet
**ArXiv:** [2211.12433](https://arxiv.org/abs/2211.12433) / [2209.03952](https://arxiv.org/abs/2209.03952)
**Venue:** ICASSP 2023 / IEEE TASLP 2023

**How it works:**
- Operates in T-F domain (complex spectrogram)
- Three types of processing per block: (1) intra-frame spectral (within each time frame), (2) sub-band temporal (across time within a frequency band), (3) full-band self-attention (global frequency context)
- Predicts complex mask; applies to mixture STFT; adds novel summation-consistency loss
- Uses BLSTM for temporal modeling (SPMamba replaces this with Mamba)

**Results:** 23.4–23.5 dB SI-SDRi on WSJ0-2mix (SOTA on that benchmark 2022–2024)
**Model size:** ~14M parameters (full model)
**Training data:** WSJ0-2mix (30 hours), with/without dynamic mixing
**Limitation:** Computationally heavy; BLSTM has quadratic scaling; no generative capability; not inherently speaker-conditioned

**Best efficiency variant:** SepReformer-B — 14.2M params, 39.8 G/s MACs, comparable SI-SDRi (NeurIPS 2024)
**Best Mamba variant:** SPMamba (arXiv 2404.02063) — replaces BLSTM with bidirectional Mamba; linear complexity; SOTA on Echo2Mix (SDRi 16.1 dB, SI-SNRi 15.3 dB)
**TIGER (ICLR 2025, arXiv 2410.01469):** Reduces TF-GridNet params by 94.3% and MACs by 95.3%; <1M parameters; first near-SOTA sub-1M model; introduces EchoSet realistic benchmark.

---

### B. Generative Waveform / Diffusion: SGMSE+, CDiffuSE

**Best paper:** SGMSE+ (Score-based Generative Models for Speech Enhancement)
**Repo:** [github.com/sp-uhh/sgmse](https://github.com/sp-uhh/sgmse)

**How it works:**
- Score-based SDE in complex STFT domain
- Forward process: linearly interpolates noisy speech with Gaussian noise (Ornstein-Uhlenbeck-like SDE with drift toward noisy speech)
- Reverse process: neural score model (NCSN++ / DNN backbone) iteratively removes noise
- Complex-valued STFT as representation
- Key advantage over pure diffusion: conditioning on noisy speech in the forward process makes reverse process much easier (the model doesn't start from pure noise)

**Results (speech enhancement):** Strong WB-PESQ, SI-SDR; best perceptual quality at time of publication; outperforms discriminative models on naturalness metrics
**Limitation for separation:** Designed for enhancement (single speaker); extension to multi-speaker requires separate conditioning approach; slow inference (multiple SDE steps); SI-SDR typically lower than discriminative masking models

**DiffSep:** First diffusion-based multi-speaker separation extension
**Latest (2025):** DiTSE (arXiv 2504.09381) — Latent Diffusion Transformer for high-fidelity generative speech enhancement; Geneses (2601.18456) — flow matching SE+SS using latent VAE space + SSL conditioning

---

### C. Generative Token / Codec LLM

**Paradigm papers:**

**TSELM (arXiv 2409.07841, ICASSP 2025):**
- Uses multiple WavLM discretized layer tokens as input
- Cross-attention to integrate enrollment speaker information
- Language model predicts token sequence
- Cross-entropy loss: converts regression → classification over codec codebook
- HiFi-GAN reconstructs waveform from predicted tokens
- Achieves excellent speech quality; comparable intelligibility to discriminative baselines
- Code: [github.com/Beilong-Tang/TSELM](https://github.com/Beilong-Tang/TSELM)

**GenTSE (arXiv 2512.20978, Dec 2024):**
- Two-stage decoder-only LM: Stage 1 predicts coarse semantic tokens; Stage 2 generates fine acoustic tokens
- Uses continuous SSL + codec embeddings (richer than discrete-only)
- Frozen-LM Conditioning (FLC) to reduce exposure bias (dWER 0.217 → 0.172)
- Direct Preference Optimization (DPO) aligns output with human perceptual preferences
- Evaluated on Libri2Mix; surpasses prior LM-based systems in quality, intelligibility, speaker consistency
- Higher perceptual quality + speaker similarity than discriminative; discriminative still wins on dWER

**LauraTSE (arXiv 2504.07402, Apr 2025):**
- Auto-regressive decoder-only LM for TSE
- Compact AR-LM predicts coarse target speech representations conditioned on continuous mixture/reference
- Lightweight encoder-only LM recovers fine-grained acoustic details
- Discriminative–Generative framework (2601.06006): front-end discriminative TSE → generative codec-LM back-end for quality refinement

**UniSep (arXiv 2503.23762, ICME 2025):**
- LLM-based universal audio separation (speech + music + sound)
- Sequence-to-sequence on discrete audio tokens from pre-trained codec
- 36.5k hours training (speech + music + sound)
- Novel audio-only pre-training reduces synthetic data dependency
- First LLM-based universal audio separation system
- Competitive with single-task specialist models

**Key limitation of codec-LM paradigm:** Slower inference than discriminative models; hallucination risk; poor SI-SDR scores (the metric is ill-suited); WER/speaker similarity are better evaluation axes.

---

### D. Flow Matching (VoiceBox-style)

**Best representative:** FlowSep (ICASSP 2025, arXiv 2409.07614) — see full summary above.

**Additional:**
- **Geneses** (arXiv 2601.18456): Latent flow matching with SSL-conditioned diffusion Transformer for SE+SS
- **AnyEnhance** (arXiv 2501.15417): Masked generative model (not strictly flow matching, but related paradigm) — two-stage: semantic enhancement → acoustic token prediction; prompt-guidance for in-context TSE; self-critic mechanism; handles speech + singing; all enhancement tasks in one model; outperforms specialized models
- **FlowTSE / MeanFlow-TSE / AD-FlowTSE** (2025): Speaker-conditioned flow matching for TSE specifically

**Advantages over diffusion:** Fewer inference steps, straighter trajectories, less mode-dropping; but perceptual vs. SI-SDR tradeoff still applies vs. discriminative masking.

---

### E. LLM Prompting / Instruction-Following Separation

**Best paper: SALMONN (ICLR 2024, arXiv 2310.13289):**
- Dual encoder: Whisper (speech/phonetic) + BEATs (audio event)
- Window-level Q-Former connects variable-length audio to LLM
- LLM backbone: Vicuna-13B
- Accepts text prompts or spoken commands
- Capabilities: ASR, emotion recognition, audio captioning, music analysis
- **Speech separation capability:** Limited — can describe a mixture but cannot cleanly output separated waveforms; it outputs text, not audio

**Qwen2-Audio (arXiv 2407.10759, July 2024):**
- Single Whisper-large encoder + Qwen2-7B LLM
- Outperforms Gemini-1.5-pro on AIR-Bench
- Voice instruction following, multilingual ASR, music analysis
- Same limitation: outputs text, not separated audio

**Key insight:** Current LLM-prompting models (SALMONN, Qwen-Audio, Gemini Audio, GPT-4o Audio) excel at *describing* what's in a mixture and performing ASR on individual speakers when prompted, but they do not natively output separated waveforms. The codec-LLM paradigm (UniSep, GenTSE, TSELM) is the line of work that produces actual audio output.

**Emerging bridge:** GenTSE + LauraTSE use LM-style decoding but produce audio tokens → waveform. This is the real "LLM for audio separation" direction.

---

## PART III: SPECIFIC QUESTIONS ANSWERED

---

### Q1. Is there any 2025 model doing TSE with >20 dB SI-SDR AND handling music AND multilingual?

**Short answer: No single model currently does all three simultaneously.**

Here is where each criterion stands:

**>20 dB SI-SDR on speech benchmarks:**
- SepReformer-L: 25.1–25.4 dB SI-SNRi on WSJ0-2mix (NeurIPS 2024)
- TF-GridNet: 23.4–23.5 dB SI-SDRi (ICASSP 2023)
- USEF-TFGridNet: 23.3 dB SI-SDRi in noisy+reverberant conditions
- MossFormer2: SOTA-class on WSJ0-2/3mix, WHAM!, WHAMR!
- These are achieved on *English-only* clean WSJ0 mixtures

**Music separation (>SDR 10 dB):**
- TFC-TDF-UNet v3 and HTDemucs handle music (MDX23 winners)
- These are not multilingual speech models
- Vocal SDR on MUSDB18 reaching ~9–10 dB with top systems

**Multilingual TSE:**
- "Leveraging Language Information for Target Language Extraction" (arXiv 2511.01652, Nov 2025): First multilingual target *language* extraction model; +1.22 dB SI-SNR improvement; uses multilingual SSL pre-trained model
- Miipher-2: 300+ language restoration, but restoration not separation; no 20 dB SI-SDR claim

**The gap:** No published model simultaneously achieves: (a) >20 dB SI-SDR, (b) music domain, (c) multilingual speech. This is a clear open research direction. The combination would require: multilingual SSL speaker encoder + multi-domain training data + a TF-GridNet/SepReformer-class separator.

---

### Q2. Best Approach for Preserving Speaker Identity (Speaker Cosine Similarity, not just SI-SDR)

**The problem:** Discriminative masking maximizes SI-SDR but can distort timbre, introduce spectral artifacts, or suppress target speaker high-frequency components. Generative models tend to produce more natural-sounding speech but can hallucinate or alter prosody.

**Best approaches for speaker identity preservation:**

1. **Generative codec-LLM (GenTSE, LauraTSE):** Explicitly optimized with speaker consistency loss in codec token space. DPO in GenTSE aligns with human perceptual preferences. GenTSE outperforms discriminative methods on speaker similarity (SECS).

2. **Discriminative-Generative hybrid (arXiv 2601.06006):** Discriminative front-end provides stable extraction; generative back-end (codec LM) restores fine-grained speaker characteristics lost by masking. This combination achieves the best tradeoff.

3. **Miipher-style conditioning:** Using speaker + linguistic features as joint constraints; preserves identity because the model is conditioned on both the speaker's voice and the transcript

4. **AnyEnhance with prompt-guidance:** Reference audio prompt guides the masked generative model to preserve target speaker timbre

**Metrics used (2024 standard):**
- **Speaker Encoder Cosine Similarity (SECS):** Computed between generated speech and ground truth using WavLM-base-plus-sv or ERes2Net-large embeddings
- **ECAPA-TDNN cosine similarity:** Also common
- Supplemented by DNSMOS, NISQA, UTMOS for naturalness
- dWER (word error rate using extracted speech) for intelligibility

---

### Q3. Best Speaker Encoders in 2024–2025 TSE

**Evolution of speaker encoder choice:**

| Era | Dominant Choice | Limitation |
|---|---|---|
| 2018–2020 | d-vector (GE2E, Google) | Shallow 3-layer LSTM; limited capacity |
| 2020–2022 | x-vector (TDNN) | Better but still misses fine spectral detail |
| 2021–2023 | ECAPA-TDNN | Best discriminative speaker encoder; used in VoxCeleb leaderboard; lowest EER (1.71%) |
| 2023–2025 | **WavLM-based** (microsoft/wavlm-base-sv, wavlm-large) | Best overall; captures both phon