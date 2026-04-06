I now have comprehensive data across all topics. Let me compile the definitive report.

---

# Production-Grade Target Speaker Extraction: Comprehensive Research Report

*Project context: TTSE-Net (TTS-guided Target Speaker Extraction), April 2026*

---

## SECTION 1: TRAINING DATASETS — SPEECH

### 1.1 Large-Scale Multilingual Speech Corpora

---

#### VoxPopuli (Meta AI, arXiv 2101.00390)
- **Size:** 400,000 hours unlabeled speech; 1,800 hours transcribed in 16 languages; 17,300 hours oral interpretation pairs (15 target languages)
- **Languages:** 23 European languages: Bulgarian, Czech, Croatian, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian, Slovak, Slovene, Spanish, Swedish
- **License:** CC0 (public domain dedication — **maximum permissiveness, commercial use fully allowed**)
- **Download:** https://github.com/facebookresearch/voxpopuli; also on Hugging Face at `facebook/voxpopuli`
- **Relevance to TTSE-Net:** Large unlabeled pool for self-supervised speaker encoder pre-training; transcribed subset for multilingual alignment. No Hindi.

---

#### Multilingual LibriSpeech (MLS, Meta AI)
- **Size:** ~50,000 hours total: ~44,500 hours English + ~6,000 hours across 7 other languages
- **Languages:** English, German, Dutch, Spanish, French, Italian, Portuguese, Polish (8 total). **No Hindi.**
- **License:** Public domain (derived from LibriVox audiobooks). No restrictions, commercial use allowed.
- **Download:** OpenSLR-94 (https://www.openslr.org/94/), AWS S3, Hugging Face `facebook/multilingual_librispeech`
- **Cost:** Free
- **Relevance:** Primary English training corpus for Stage 1–2. High speaker diversity (read speech, 44k+ English hours). Reliable, clean, well-aligned.

---

#### Mozilla Common Voice (v20, December 2024)
- **Size:** 33,150 hours validated across all languages (Common Voice 20)
- **Languages:** 133 languages including Hindi (Hindi subset: approximately 200–400 hours depending on version; check CV-20 for exact count)
- **License:** CC0 (fully open, commercial use allowed)
- **Download:** https://commonvoice.mozilla.org/en/datasets; Hugging Face `mozilla-foundation/common_voice_XX_0`
- **Cost:** Free
- **Quality note:** Crowdsourced, variable recording conditions — useful for noise robustness training. Hindi subset quality is lower than Shrutilipi/Kathbath but provides real-world microphone diversity.

---

#### GigaSpeech (SpeechColab)
- **Size:** 10,000 hours labeled (XL configuration); smaller subsets: XS=10h, S=250h, M=1,000h, L=2,500h
- **Languages:** English only (GigaSpeech v1)
- **License:** Restricted — research and educational use only. Models trained on GigaSpeech may be commercially deployed under "Fair Use" terms, but the raw data cannot be redistributed commercially. Not suitable for building commercial data pipelines.
- **Download:** Hugging Face `speechcolab/gigaspeech` (requires license agreement)
- **Cost:** Free to researchers under non-commercial terms
- **Relevance:** Good multi-domain English (YouTube, podcasts, audiobooks) with diverse speaking styles. Useful for Stage 2 noise robustness.

#### GigaSpeech 2 (2024, arXiv 2406.11546)
- **Size:** 30,000 hours raw; ~22,000 hours refined (Thai: 10k, Indonesian: 6k, Vietnamese: 6k)
- **Languages:** Thai, Indonesian, Vietnamese — **not directly relevant to TTSE-Net** but demonstrates the automated pipeline methodology
- **License:** Apache 2.0 (more permissive than GigaSpeech v1)
- **Download:** Hugging Face `speechcolab/gigaspeech2`

---

#### WenetSpeech (Chinese)
- **Size:** 10,000+ hours high-quality labeled + 2,400 hours weakly labeled + 10,000 hours unlabeled = **22,400+ hours total**
- **Languages:** Mandarin Chinese only
- **License:** CC BY 4.0 (non-commercial for raw audio; models can be commercial). Copyright remains with original creators (YouTube/podcast sources).
- **Download:** https://github.com/wenet-e2e/WenetSpeech; Hugging Face `wenet-e2e/wenetspeech`

#### WenetSpeech4TTS (2024)
- **Size:** 12,800 hours paired audio-text, derived from WenetSpeech, quality-filtered for TTS training
- **License:** CC BY 4.0
- **Hugging Face:** `Wenetspeech4TTS/WenetSpeech4TTS`
- **Relevance:** If adding Mandarin support (not in current scope but extensible).

---

#### AISHELL-3 (AISHELL Tech)
- **Size:** ~85 hours, 88,035 utterances, **218 native Mandarin speakers** (multi-speaker TTS corpus)
- **Languages:** Mandarin Chinese
- **License:** Apache 2.0 (commercial use allowed)
- **Download:** OpenSLR-93 (https://www.openslr.org/93/), ~19 GB; Hugging Face `AISHELL/AISHELL-3`
- **Relevance:** Small but high-quality multi-speaker Chinese corpus. Speaker diversity useful for speaker encoder training. Gender/age/accent metadata included.

---

#### EMILIA (2024, arXiv 2407.05361) — **Highly Recommended**
- **Size:** 101,654 hours of in-the-wild speech (original); **Emilia-Large = 215,000 hours** (101k Emilia + 114k Emilia-YODAS)
- **Languages:** English, Chinese, German, French, Japanese, Korean (6 languages); 24 kHz audio
- **Source:** Scraped from diverse internet sources (talk shows, interviews, debates, sports commentary, audiobooks, podcasts) — much more naturalistic than read-speech corpora
- **License:**
  - Original Emilia 101k hours: **CC BY-NC 4.0** (non-commercial only)
  - Emilia-YODAS 114k hours: **CC BY 4.0** (commercial use allowed)
  - Emilia-Large (combined): mixed license
- **Download:** Hugging Face `amphion/Emilia-Dataset`; also OpenDataLab
- **Relevance:** Best available corpus for training on spontaneous/expressive speech. The in-the-wild nature is critical for TSE training on podcast/broadcast scenarios. Use Emilia-YODAS portion for commercial products.

---

#### LibriLight (Meta AI)
- **Size:** 60,000 hours unlabeled English speech; labeled subsets: 10h, 1h, 10min
- **Speakers:** 7,000+ unique speakers
- **Source:** LibriVox audiobooks (English)
- **License:** MIT license (code); audio inherits LibriVox public domain
- **Download:** Direct: `https://dl.fbaipublicfiles.com/librilight/data/large.tar` (warning: large.tar is multi-terabyte); GitHub `facebookresearch/libri-light`
- **Relevance:** Massive unlabeled pool for self-supervised speaker encoder pre-training. GPU-hours needed to process 60k hours; plan for ~100TB storage.

---

#### LDC Corpora (Fisher, Switchboard, CallHome)

**Access model:** LDC membership $2,400/year (standard) grants 16 corpora of choice. Non-member per-corpus pricing varies ($300–$1,500 per corpus depending on age and type).

| Corpus | LDC ID | Content | Hours (approx.) | Notes |
|--------|--------|---------|-----------------|-------|
| Fisher English Part 1 (Speech) | LDC2004S13 | Telephone conversational speech | ~900h | 2-speaker, natural conversation |
| Fisher English Part 2 (Speech) | LDC2005S13 | Telephone conversational speech | ~900h | Continuation of Fisher |
| Switchboard-1 Release 2 | LDC97S62 | Telephone speech | ~260h | Classic benchmark corpus |
| CallHome American English | LDC97S42 | Telephone family calls | ~17h | Spontaneous, casual speech |

- **License:** LDC User Agreement — research/education; commercial use requires explicit negotiation
- **Purchase path:** Visit https://catalog.ldc.upenn.edu/ → Join LDC (academic: ~$2,400/yr, for-profit: see LDC2024MFP) → Select corpora. For-profit membership rates are significantly higher.
- **Recommendation for TTSE-Net:** Fisher is the highest-value LDC corpus — 1,800 hours of conversational 2-speaker telephone speech directly applicable to multi-speaker TSE training.

---

#### Commercial Data Vendors

| Vendor | Offerings | Notes |
|--------|-----------|-------|
| **Appen** | 500+ locales, TTS synthesis, ASR transcription, acoustic event labeling | Production-grade; custom collection for underrepresented languages including Hindi |
| **Lionbridge AI** | Custom speech collection, annotation, Aurora AI Studio | Strong in low-resource languages; launched ML platform 2024 |
| **MagicData** | 400+ datasets, 60+ languages/dialects; commercial licensing | Primarily Mandarin; has Indic language data. Contact: business@magicdatatech.com |
| **Shaip** | Indian language datasets; Hindi, Tamil, Telugu, etc. | Specialized in Indian languages; https://www.shaip.com/solutions/indian-language-datasets/ |

*Note: SoundHound is primarily a voice AI product company, not a dataset vendor.*

---

### 1.2 Hindi-Specific Datasets

---

#### Shrutilipi (AI4Bharat, IIT Madras)
- **Size:** 6,400+ hours across **12 Indian languages**; Hindi subset: ~**1,600 hours**
- **Source:** Mined from All India Radio (AIR) news bulletins — read speech, professional quality
- **Quality:** High — broadcast-quality audio, professionally read; however domain is limited to news
- **License:** **CC BY 4.0** (commercial use allowed)
- **Format:** fairseq-compatible (audio paths + transcripts)
- **Download:** Hugging Face `ai4bharat/Shrutilipi`; https://datasets.ai4bharat.org/shrutilipi/
- **Known issues:** Domain limited to formal news reading; limited speaker diversity; some segments have background music (AIR production). Quality filtering recommended before use.

---

#### Kathbath / IndicSUPERB (AI4Bharat, arXiv 2208.11761)
- **Size:** 1,684 hours total across **12 Indian languages** (published at AAAI 2023)
- **Speakers:** 1,218 contributors from 203 districts across India
- **Tasks covered:** ASR, speaker verification, speech identification, query by example, keyword spotting
- **Format:** m4a audio files
- **License:** CC BY 4.0
- **Download:** Hugging Face `ai4bharat/Kathbath`; GitHub `AI4Bharat/IndicSUPERB`
- **Relevance:** Primary Hindi speaker verification/identification benchmark; Hindi ASR data. The geographic spread (203 districts) gives accent diversity critical for multilingual TSE.

---

#### Vistaar (AI4Bharat, INTERSPEECH 2023)
- **Nature:** **Benchmark + training set collection** (not a single dataset)
- **Size:** 59 benchmarks across 12 Indian languages; 10,700+ hours training data total across language/domain combinations
- **Domains covered:** News, blogs, education, weather, books, navigation, tourism, agriculture, poems, stories, science, health
- **Benchmark sources:** Kathbath, FLEURS, CommonVoice, IndicTTS, MUCS, GramVaani
- **WER on Hindi:** 13.6% (best system: IndicWhisper fine-tuned on Vistaar train data) — the **lowest WER of any language** in the benchmark, indicating relatively more data
- **Download:** GitHub `AI4Bharat/vistaar`
- **Use case for TTSE-Net:** Use Vistaar Hindi train sets as a combined Hindi ASR training pool; use Vistaar benchmarks for evaluating Hindi intelligibility (WER post-extraction).

---

#### MUCS 2021 (Multilingual and Code-Switching ASR Challenge)
- **Hindi-English Code-Switched subset:**
  - Train: **89.86 hours**, 16 kHz, 16-bit
  - Dev (hidden): 6.24 hours (4,034 segments)
  - Test: 5.18 hours
  - Total: ~101 hours
- **Source:** Spoken tutorials on computer science topics (IIT Bombay origin)
- **Vocabulary:** 17,877 words (train set)
- **File size:** Train = 7.3 GB; Test = 443 MB
- **Download:** OpenSLR-104 (https://www.openslr.org/104/)
- **Relevance:** **The primary Hindi-English code-switched corpus.** Essential for the code-switched test track of the TTSE-Net benchmark. Limited to technical domain (CS tutorials) — will not cover casual code-switching.

---

#### IndicVoices-R (AI4Bharat, NeurIPS 2024)
- **Size:** 1,704 hours, **22 Indian languages**, 10,496 speakers
- **Speech type:** 93.25% extempore (spontaneous) speech — far more expressive than read-speech corpora
- **Quality:** Comparable to LJSpeech/LibriTTS (verified by MOS scores)
- **Per-language range:** 9–175 hours per language
- **Purpose:** Designed specifically for multi-speaker TTS training
- **Download:** Hugging Face `ai4bharat/indicvoices_r`
- **Relevance to TTSE-Net:** Excellent for Hindi TTS enrollment synthesis (multi-speaker, spontaneous, high-quality). Use as source for training Hindi zero-shot TTS systems.

---

#### CVIT IndicSpeech (IIT Hyderabad / IITB)
- **Hindi subset:** 24 hours of Hindi TTS data (single speaker, high quality)
- Also: Malayalam 24h, Bengali 24h
- **Use case:** Small but high-quality Hindi speech for TTS fine-tuning

---

#### MLS Hindi
- Multilingual LibriSpeech does not include Hindi — **MLS does not have a Hindi component**. The FINAL_RESEARCH_PLAN.md reference to "MLS Hindi" is incorrect; use Shrutilipi/Kathbath instead.

---

#### AI4Bharat Ecosystem Summary for Hindi

| Dataset | Hours (Hindi) | Type | License | Best Use |
|---------|---------------|------|---------|----------|
| Shrutilipi | ~1,600 | ASR (news, read) | CC BY 4.0 | ASR training, speaker diversity |
| Kathbath | ~140 (Hindi portion) | Multi-task eval + train | CC BY 4.0 | Benchmark + speaker verification |
| IndicVoices-R | ~100–175 | Multi-speaker TTS | Check HF card | Hindi TTS enrollment synthesis |
| MUCS 2021 | ~90 (code-switched) | Code-switched ASR | OpenSLR | Code-switched TSE test set |
| Vistaar (aggregate) | ~800+ | Multi-domain ASR | Per-source | ASR training pool |
| Common Voice 20 | ~200–400 | Crowdsourced ASR | CC0 | Speaker diversity, noise robustness |

**Total accessible Hindi speech: approximately 2,000–3,000 hours** with reasonable licensing.

---

### 1.3 Noise and Music Datasets

---

#### MUSDB18 vs MUSDB18-HQ

| Feature | MUSDB18 | MUSDB18-HQ |
|---------|---------|------------|
| Tracks | 150 full-length songs (100 train, 50 test) | Same 150 tracks |
| Format | Native Instruments STEMS (.mp4), AAC @256kbps | Uncompressed WAV |
| Bandwidth | ~16 kHz (AAC compression limit) | Up to 22 kHz |
| Duration | ~10 hours total | ~10 hours total |
| Stems | Drums, bass, vocals, other + mixture | Same 4 stems |
| License | **CC BY-NC-SA 4.0** (non-commercial only) | **CC BY-NC-SA 4.0** (non-commercial only) |
| Download | Zenodo record 1117372 | Zenodo record 3338373 |

**Critical note:** Both MUSDB versions are **non-commercial**. Use MUSDB18-HQ for research/paper; for any commercial product pipeline, substitute MUSAN music (CC BY) or licensed commercial music.

**Recommendation for TTSE-Net:** Use MUSDB18-HQ for ablation study and benchmark test set (research context). For Stage 3 training (music backgrounds), supplement with MUSAN music track (CC BY 4.0, ~42 hours of music).

---

#### FreeSound
- **Content:** 500,000+ sounds; diverse mix of sound effects, ambience, music fragments
- **Licenses:** Mixed — CC0, CC BY, CC BY-NC, CC BY-SA. Filter with API for commercial-safe subsets.
- **Batch Download:**
  - API v2: https://freesound.org/docs/api/
  - Rate limit: **500 full-quality downloads/day** (OAuth2 required for full files; no auth needed for 128kbps previews)
  - Strategy: Register API key → use `filter=license:("Creative Commons 0" OR "Attribution")` parameter → automate with python `freesound` library
  - PyPI: `pip install freesound`
  - Note: For large-scale download, expect 2–3 weeks at 500/day limit

---

#### AudioSet (Google)
- **Content:** ~2.1 million 10-second YouTube clips, 632 audio classes, weakly labeled
- **Direct audio:** **Google does NOT distribute audio directly** — only pre-computed log-mel embeddings (128-dim, 1Hz)
- **How to get raw audio:** Must download via YouTube IDs using yt-dlp
  - Tools: `audioset-download` (PyPI), `audioset-processing` (GitHub)
  - **Expected attrition:** 20–40% of clips are no longer available on YouTube
  - Pre-downloaded version: Hugging Face `agkphysics/AudioSet` (March 2023 snapshot, not complete)
- **License:** YouTube ToS applies to individual clips; AudioSet metadata is CC BY 4.0
- **Practical recommendation:** Use the pre-downloaded Hugging Face snapshot for noise augmentation; supplement with FreeSound for reliable CC audio.

---

#### LAION-Audio-630K
- **Size:** 633,526 audio-text pairs, **4,325 hours** total
- **Sources:** 8 sources; only 4 are publicly released:
  - BBC Sound Effects (URL-linked CSV)
  - Epidemic Sound (URL-linked CSV — note: Epidemic Sound requires subscription for access)
  - Audiostock (URL-linked CSV — Japanese commercial library)
  - Freesound (released directly to Hugging Face)
- **License:** Mixed per source; the Freesound subset is CC-licensed
- **Download:** GitHub `LAION-AI/audio-dataset`; Hugging Face `marianna13/LAION-Audio-630k`
- **Relevance:** Useful for text-conditioned audio understanding; less directly relevant to TSE noise augmentation

---

#### WavCaps
- **Size:** ~400,000 audio clips with captions
- **Notable:** Uses ChatGPT-refined captions from basic audio labels
- **Limitation:** Captions are shallow reformulations of labels; not high semantic fidelity
- **Use case:** Audio-language pre-training; less relevant for TSE noise simulation

---

#### Commercial Music Libraries for ML Training

**Important context:** Standard royalty-free licenses (Epidemic Sound, Artlist) do NOT include ML training rights. You need explicit ML/AI training licenses.

| Option | Details |
|--------|---------|
| **Epidemic Sound** | 35,000+ tracks, 90,000+ SFX; owns all rights to its catalog (no PRO affiliation). Licensing for AI training requires direct negotiation — contact their enterprise team |
| **Artlist** | 18,000+ tracks; recently integrated Google Lyria AI generation. Standard license covers sync; ML training requires custom agreement |
| **MUSAN (music subset)** | 42 hours of music, CC BY 4.0, free. Specifically designed for speech research. Download: `http://www.openslr.org/17/` |
| **Free Music Archive (FMA)** | 106,574 tracks, 917 GB, CC-licensed (filter for CC0/CC BY for commercial). Download: GitHub `mdeff/fma` |
| **ccMixter** | CC-licensed music; API available for programmatic access |

**Recommended noise/music corpus stack for TTSE-Net:**
1. **MUSAN** (CC BY): 60h total: 42h music, 11h speech, 7h noise — for all training stages
2. **FMA (CC0/CC BY filtered)**: 100–200h additional music — for Stage 3+ music diversity
3. **MUSDB18-HQ** (CC BY-NC): For research benchmark only (ablation, test sets)
4. **DNS Challenge noise** (Microsoft, MIT): ~180h noise signals, diverse environments

---

## SECTION 2: SIMULATION AND MIXING TOOLS

### pyroomacoustics
- **Purpose:** Python library for audio room simulation and array processing (EPFL LCAV)
- **Core capabilities:**
  - Image Source Method (ISM) for RIR generation in rectangular (shoebox) and arbitrary polyhedral rooms
  - Hybrid ISM + Ray Tracing for capturing late reverberation and scattering
  - Variable wall absorption coefficients per wall per frequency band
  - Air absorption simulation
  - Microphone array beamforming (DAS, MVDR, DSB)
  - 2D and 3D room geometries
- **Speed:** C++ backend; fast for sequential RIR generation. For massive parallel RIR generation, consider gpuRIR (see below).
- **Usage:** `pip install pyroomacoustics`
- **Best for:** Generating diverse reverb conditions for training data augmentation. Key parameters: room dimensions (2–20m), RT60 (0.2–1.5s), source/mic positions.

### gpuRIR
- **Purpose:** GPU-accelerated ISM-based RIR simulation
- **Speed:** ~100× faster than CPU implementations; enables batched simulation of thousands of RIRs in parallel
- **Comparison to pyroomacoustics:** When using lookup tables (LUTs), pyroomacoustics performance approaches gpuRIR on single RIRs. gpuRIR wins decisively for **batched generation** (simulating many room configurations in parallel during training data creation).
- **Install:** `pip install gpuRIR` (requires CUDA)
- **GitHub:** https://github.com/DavidDiazGuerra/gpuRIR
- **Recommendation:** Use gpuRIR for pre-generating the full RIR library for training data simulation; use pyroomacoustics for prototyping and room design.

### scaper
- **Purpose:** Soundscape synthesis and augmentation (NYU Music and Audio Research Lab)
- **Capabilities:**
  - Probabilistic specification of soundscapes ("events drawn from distribution X at times sampled from Y")
  - Per-event audio transformations: pitch shifting, time stretching, reverb
  - Generates JAMS-format metadata alongside audio (fully reproducible)
  - Batch generation of thousands of soundscapes from a single spec
- **Install:** `pip install scaper`
- **GitHub:** https://github.com/justinsalamon/scaper
- **Best for:** Generating the speech+noise+music mixtures for TTSE-Net training corpus. Can integrate with pyroomacoustics for reverberant soundscapes.

### audiomentations
- **Purpose:** CPU-based audio data augmentation for deep learning (Iver Jordal)
- **Augmentations include:** AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, RoomSimulator, AddBackgroundNoise, AddShortNoises, LoudnessNormalization, HighPassFilter, LowPassFilter, BandPassFilter, and 30+ others
- **Install:** `pip install audiomentations`
- **Strength:** Production-tested, well-documented, supports multi-channel audio
- **Best for:** Online augmentation in training dataloader

### torch-audiomentations
- **Purpose:** GPU-accelerated version of audiomentations for PyTorch
- **Key feature:** Processes batches of audio tensors on GPU — eliminates CPU↔GPU transfer bottleneck for augmentation
- **Caveat:** GPU audio augmentation is not always faster than CPU for small batches (GPU kernel launch overhead); breakeven point is typically batch size > 32 with long audio clips
- **Install:** `pip install torch-audiomentations`
- **GitHub:** https://github.com/asteroid-team/torch-audiomentations

### Recommended Stack for Speech+Music+Noise Mixture Synthesis

```
Stage 1: Room simulation
  - gpuRIR → pre-generate 50,000+ RIRs covering 3D room distribution
  - Parameters: RT60 ∈ [0.1, 1.5s], room 2×2×2m to 15×15×4m

Stage 2: Mixture generation (offline, before training)
  - scaper: schedule speech sources (2–4 speakers), music events, noise events
  - Apply sampled RIRs from Stage 1 to each source independently
  - Mix at target SNR: speech-to-noise ∈ [-10, 10dB]; music ∈ [-5, 15dB]
  - Output: mixture + clean target + metadata (JAMS)

Stage 3: Online augmentation (during training)
  - torch-audiomentations (on GPU): gain jitter, band filtering, codec simulation
  - audiomentations (on CPU dataloader workers): time-domain perturbations

Best tool for final mixture: scaper + gpuRIR combination
```

---

## SECTION 3: LARGE MODEL TRAINING INFRASTRUCTURE

### 3.1 Hardware Requirements for 1B+ Parameter Audio Models

| Model Scale | Minimum VRAM per GPU | Recommended | Notes |
|-------------|---------------------|-------------|-------|
| 100M params | ~8 GB | A40/A100 40GB | Fits single GPU |
| 300M params | ~24 GB (FP16) | A100 40GB | Audio processing overhead adds ~20–30% vs LLM |
| 1B params | ~80 GB (FP16) | A100 80GB × 2–4 | With optimizer states (AdamW): ×4 multiplier |
| 1B params (ZeRO-3) | ~20 GB/GPU | A100 80GB × 4 | Optimizer sharded across 4 GPUs |
| 3B+ params | Multi-node | H100 80GB × 8+ | 8× H100 SXM per node is standard |

**Memory breakdown for 1B parameter model (FP32 training):**
- Parameters: 4 GB
- Gradients: 4 GB
- AdamW optimizer states (fp32 master): 8 GB
- Activations (audio is sequence-heavy): 20–40 GB (function of context length)
- **Total: ~36–56 GB** — fits within 2× A100 80GB with standard parallelism

**For TTSE-Net (34M params):** Single A100 40GB is sufficient. For 1B parameter backbone experiments, use 4–8× A100 80GB.

---

### 3.2 FSDP vs DeepSpeed for Audio Models

| Feature | PyTorch FSDP | DeepSpeed ZeRO-3 |
|---------|-------------|------------------|
| Best range | 100M–1B parameters | 1B–100B+ parameters |
| Iteration speed | Up to **5× faster** than ZeRO-3 in 100M–1B range | Slower per iteration at medium scale |
| Memory efficiency | Good (FULL_SHARD mode) | Superior at extreme scale |
| CPU offload | All-or-nothing only | Granular (params + optimizer separately); NVMe offload supported |
| Ease of integration | Native PyTorch, minimal config | Requires `deepspeed` config JSON; more complex |
| Communication overlap | Manual `--fsdp_auto_wrap_policy` needed | Transparent |
| Hugging Face Accelerate | First-class support | First-class support |

**Recommendation for TTSE-Net:**
- **For TTSE-Net (34M):** Standard DDP (DistributedDataParallel) is sufficient; FSDP unnecessary
- **For 1B+ backbone experiments:** Use **FSDP** (simpler, faster at this scale)
- **For >3B models or memory-constrained clusters:** Switch to **DeepSpeed ZeRO-3**
- Both are configurable via Hugging Face `accelerate` with minimal code changes

---

### 3.3 Multi-Node Training Best Practices for Audio Models

1. **Network fabric:** Use InfiniBand (HDR/NDR) for inter-node communication; Ethernet will bottleneck gradient synchronization. NCCL backend requires IB for efficient all-reduce.

2. **Data loading:** Audio models have high I/O demands. Use:
   - WebDataset format (tar shards) for efficient streaming
   - Local NVMe SSD caches for hot datasets; avoid NFS for training data
   - Pre-compute mel spectrograms or encodings; avoid on-the-fly computation at training time for large datasets

3. **Communication overlap (CO2 technique, 2024–2025):** Use dual-buffer pipeline to overlap gradient communication with computation — critical for reducing inter-node latency impact. Available via Megatron-LM or custom DDP hooks.

4. **Gradient checkpointing:** Essential for audio transformers with long sequences. PyTorch `checkpoint_sequential()` or HuggingFace `gradient_checkpointing_enable()`.

5. **Mixed precision:** BF16 preferred over FP16 for audio models (better numerical stability for the dynamic range of audio features). Use `torch.autocast("cuda", dtype=torch.bfloat16)`.

6. **Single-node vs multi-node note:** 4 GPUs on 1 node is faster than 1 GPU on 4 nodes — inter-node bandwidth is always the bottleneck. Maximize single-node occupancy before going multi-node.

7. **Checkpoint strategy:** Save every 1,000 steps minimum; use async checkpointing to avoid blocking training. Consider Megatron-LM distributed checkpointing for models >1B.

---

### 3.4 Pre-trained Backbone Options

| Backbone | Params | Pre-train data | Best downstream tasks | TTSE-Net relevance |
|----------|--------|---------------|----------------------|-------------------|
| **WavLM-Large** | 317M | 94k hours (LibriLight + others) | Speaker verification (EER 0.383% on VoxCeleb1), speaker ID, diarization | **Top choice for speaker encoder** — state-of-the-art on all speaker tasks |
| **wav2vec2-large** | 317M | 60