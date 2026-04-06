I now have comprehensive information for all models. Let me compile the full report.

---

# SAM Audio and Related Audio Foundation Models: Comprehensive Research Report

*Research conducted April 2026. Covers models through early 2026.*

---

## PART 1: What Is "SAM Audio"? — The Critical Disambiguation

**SAM Audio = "Segment Anything in Audio" by Meta FAIR.** This is the primary and definitive meaning of the term as of late 2025. There is no separate speech AI startup or commercial product called "SAM Audio" — all roads lead to Meta's December 2025 model release. If someone in a speech AI context says "SAM Audio," they almost certainly mean Meta's model.

- It is **not** "Stable Audio" by Stability AI (that is a text-to-audio generation model, unrelated to separation)
- There is no prior "AudioSAM" academic paper that predates Meta's; the closest is AV-SAM (arXiv 2305.01836), which applied SAM to audio-visual segmentation, not source separation
- A paper titled "When Denoising Hinders: Revisiting Zero-Shot ASR with SAM Audio" (arXiv 2603.04710, 2026) references Meta's SAM Audio in the context of ASR preprocessing — confirming adoption in the speech community

---

## PART 2: SAM Audio (Meta FAIR) — Detailed Profile

**Full Name:** SAM Audio: Segment Anything in Audio
**Authors:** Bowen Shi, Andros Tjandra, John Hoffman, Helin Wang, Yi-Chiao Wu, Luya Gao, Julius Richter, Matt Le, Apoorv Vyas, Sanyuan Chen, Christoph Feichtenhofer, Piotr Dollár, Wei-Ning Hsu, Ann Lee
**Year:** December 2025 (announced Dec 16, 2025; arXiv submitted Dec 19, 2025)
**ArXiv:** [2512.18099](https://arxiv.org/abs/2512.18099)
**Affiliation:** Meta FAIR

### Architecture
- **Core model:** Diffusion Transformer (DiT) trained with flow matching
- **Latent representation:** DAC-VAE encoding; audio compressed to T×128 at 25 Hz frame rate
- **Three model sizes:**
  - Small: not explicitly parameterized in public docs
  - Base: ~500M parameters (12 layers, 1,536 attention dim)
  - Large: ~1B–3B parameters (16–22 layers, 2,048–2,816 attention dim)
  - *Note: parameter counts exclude external encoders (PE visual encoder, T5 text encoder, DAC-VAE codec)*
- **Text conditioning:** T5-Base (768-dim), injected via cross-attention
- **Visual conditioning:** Perception Encoder Audio-Visual (PE-AV), frame-aligned features concatenated with audio latents
- **Span conditioning:** Frame-synchronous token sequences from time-interval annotations
- **Joint prediction:** Simultaneously generates target stem and residual stem

### Prompting Modalities (all three usable independently or jointly)
1. **Text prompting** — free-form language, e.g., "woman speaking", "violin", "dog barking"
2. **Visual prompting** — binary masks from SAM 2; click on an object in video to isolate its sound
3. **Span prompting** — mark temporal intervals; described as an "industry first" for audio separation

Additionally, a **span prediction** module (using PE-A-Frame) can automatically detect sound event boundaries to augment text prompts.

### Training Data
- ~1M hours medium-quality audio-video data
- ~21,910 hours conversational speech
- ~20,000 hours clean music
- ~10,600 music compositions (536 hours) with instrument stems
- ~10,000 hours high-quality sound effects
- Pseudo-labeled data generated via model-as-data-engine (filtered with CLAP similarity + aesthetic scoring)
- PE-AV trained on 100M+ videos via large-scale multimodal contrastive learning

### Tasks Supported
- Speech separation / cleaning (multi-speaker, single-channel)
- Speaker separation by attribute (e.g., gender)
- Music separation (vocals, instruments, stems)
- Instrument separation (37 instrument classes)
- General sound separation
- Works on both in-the-wild recordings and professionally produced audio

### Target Speaker Extraction Capability
**Yes, partially.** The model can separate speech by speaker attributes described in text (e.g., "woman speaking," "male voice") and can isolate a visually indicated speaker from video. However, it does **not** support audio-enrollment-based extraction (i.e., you cannot give it a reference clip of a target speaker and say "extract this person"). This is a key limitation compared to classical TSE systems.

### Evaluation Framework
- **SAM Audio-Bench:** First in-the-wild audio separation benchmark; sources from AudioSet, VGGSound, MUSIC, AVSpeech, CondensedMovies
- **SAM Audio Judge (SAJ):** Reference-free evaluation model scoring 9 perceptual dimensions (recall, precision, faithfulness, overall quality, plus difficulty metrics)
- Outperforms specialist models including MossFormer2, Tiger, FastGeCo across all categories

### Known Limitations
- Cannot use audio as a prompt (no speaker enrollment/reference audio)
- Struggles with highly similar sound sources (single singer in choir, one instrument in orchestra)
- RTF ≈ 0.7 (faster than real-time on appropriate hardware)

### Open Source Status
**Yes — code and checkpoints available.**
- GitHub: [facebookresearch/sam-audio](https://github.com/facebookresearch/sam-audio)
- Hugging Face: `facebook/sam-audio-base`, `facebook/sam-audio-large`, `facebook/sam-audio-judge`
- License: "SAM License" (Meta's proprietary research license, not fully permissive)
- Demo: [segment-anything.com](https://segment-anything.com)

---

## PART 3: Stable Audio (Stability AI)

**Full Name:** Stable Audio Open
**Authors:** Zach Evans, Julian D. Parker, CJ Carr, Zack Zukowski, Josiah Taylor, Jordi Pons
**Year:** 2024
**ArXiv:** [2407.14358](https://arxiv.org/abs/2407.14358)

### Architecture
- **Autoencoder:** Variational autoencoder (VAE) with latent size 64; compresses 44.1 kHz stereo to 21.5 Hz latent rate; strided convolutional encoder/decoder with dilated ResNet blocks and Snake activations
- **Generative model:** DiT operating in latent space; stacked cross-attention + gated MLP blocks with rotary positional embeddings
- **Text conditioning:** T5 text encoder (not CLAP, unlike commercial Stable Audio 2.0)
- **Timing conditioning:** Variable-length generation up to 47 seconds via timing tokens
- **Total parameters:** ~1.21 billion
- **Training data:** ~486,000 recordings — 472,618 from Freesound + 13,874 from Free Music Archive (FMA); all CC-0, CC-BY, or CC-Sampling+ licensed
- **Output:** Variable-length stereo audio at 44.1 kHz

### Can It Do Separation?
**No.** Stable Audio is a text-to-audio *generation* model. It does not perform source separation. Its purpose is generating sound effects and music from text descriptions. The name similarity to "SAM Audio" is purely coincidental.

### Open Source
**Yes.** Weights on Hugging Face (`stabilityai/stable-audio-open-1.0`), code on GitHub (`Stability-AI/stable-audio-tools`).

---

## PART 4: UniAudio

**Full Name:** UniAudio: An Audio Foundation Model Toward Universal Audio Generation
**Authors:** Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Zhou Zhao, Xixin Wu, Helen Meng
**Year:** October 2023 (ICML 2024)
**ArXiv:** [2310.00704](https://arxiv.org/abs/2310.00704)

### Architecture
- **Tokenization:** Residual Vector Quantization (RVQ) neural codec; 3 codebook layers per frame; phoneme, MIDI, text (T5), and semantic (HuBERT k-means) tokens for conditioning
- **Multi-scale Transformer:** Two-tier design to handle extreme sequence lengths
  - Global Transformer: 24 layers, 1,536 attention dim, 12 heads — **744M parameters**
  - Local Transformer: 8 layers, same specs — **238M parameters**
  - **Total: ~1 billion parameters**
- **Training:** 165,000 hours across 12 datasets (LibriLight 60K hrs, MLS 20K hrs, AudioSet 5.8K hrs, WavCaps 7K hrs, Million Song 7K hrs, others)
- **Next-token prediction** over concatenated source-target sequences

### 11 Supported Tasks
Training stage: TTS, Voice Conversion, Speech Enhancement, **Target Speaker Extraction (TSE)**, Singing Voice Synthesis, Text-to-Sound, Text-to-Music
Fine-tuning stage: Audio Edit, Speech Dereverberation, Instructed TTS, Speech Edit

### Target Speaker Extraction Capability
**Yes, explicitly supported.** Conditioning: mixed speech + 3-second speaker audio prompt. The model receives a short reference clip and extracts that speaker from the mixture. This is direct speaker-conditioned extraction, unlike SAM Audio which uses text/visual only.

### Open Source
**Yes.** Code on GitHub: [yangdongchao/UniAudio](https://github.com/yangdongchao/UniAudio)

**UniAudio 1.5** (arXiv [2406.10056](https://arxiv.org/abs/2406.10056), NeurIPS 2024) extends this with LLM-driven codec (LLM-Codec) enabling few-shot task learning; compresses 1s audio to 57 tokens; trained on 2,000 hours.

---

## PART 5: SALMONN

**Full Name:** SALMONN: Towards Generic Hearing Abilities for Large Language Models
**Authors:** Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Chao Zhang
**Year:** October 2023 (published at ICLR 2024)
**ArXiv:** [2310.13289](https://arxiv.org/abs/2310.13289)

### Architecture
- **Speech encoder:** Whisper-Large-v2 encoder
- **Audio encoder:** BEATs (fine-tuned for non-speech semantic audio)
- Both encoders run at 50 Hz; outputs concatenated frame-by-frame
- **Bridge:** Window-level Q-Former; L=17 frames per window, N=1 query token → ~88 text tokens for 30s audio
- **LLM:** Vicuna-13B (LLaMA-based)
- **Fine-tunable params:** ~33M (Q-Former + LoRA rank-8 on Vicuna attention layers); ~0.24% of total
- **Training data:** ~4,400 hours, ~2.3M samples (LibriSpeech, GigaSpeech, WavCaps, AudioCaps, Clotho, CoVoST2, IEMOCAP, MusicCaps, LibriMix, VoxCeleb1)

### Supported Tasks (15)
ASR, speech translation (En→Zh), audio captioning, phone recognition, emotion recognition, music captioning, overlapped speech recognition, speaker verification + emergent tasks: slot filling, spoken QA, storytelling, speech-audio co-reasoning, cross-lingual translation

### Target Speaker Extraction
**No direct separation output.** SALMONN is an audio-language model that describes, transcribes, and reasons about audio. It uses LibriMix in training (which involves overlapping speech) and can recognize overlapping speech, but does not output separated audio waveforms.

### Open Source
**Yes.** [github.com/bytedance/SALMONN](https://github.com/bytedance/SALMONN), model checkpoints released.

---

## PART 6: Qwen2-Audio (Alibaba Cloud)

**Full Name:** Qwen2-Audio Technical Report
**Authors:** Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, Jingren Zhou
**Year:** July 2024
**ArXiv:** [2407.10759](https://arxiv.org/abs/2407.10759)

### Architecture
- **Audio encoder:** Whisper-Large-v3 (initialized from); 128-channel mel-spectrogram at 16 kHz, 25ms window, 10ms hop, stride-2 pooling
- **LLM backbone:** Qwen-7B
- **Total parameters: 8.2B**
- **Training:** Three-stage pipeline — (1) audio-text pre-training with natural language prompts, (2) supervised fine-tuning for two interaction modes, (3) Direct Preference Optimization (DPO)
- **Two interaction modes:** Voice chat (voice-in, text-out) and Audio analysis (audio + text instruction → text response)

### Supported Tasks
ASR, speech-to-text translation, speech emotion recognition, vocal sound classification, general audio reasoning, multi-speaker understanding (can interpret audio content with multiple simultaneous speakers)

### Target Speaker Extraction
**No.** Qwen2-Audio is an audio-language understanding model that outputs text, not audio. It cannot perform source separation. The model best handles clips ≤30 seconds.

### Open Source
**Yes.** [github.com/QwenLM/Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio), checkpoints on Hugging Face (`Qwen/Qwen2-Audio-7B`, `Qwen/Qwen2-Audio-7B-Instruct`). License: Apache 2.0 / CC BY 4.0.

---

## PART 7: GAMA

**Full Name:** GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities
**Authors:** Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Kiran Reddy Evuru, Utkarsh Tyagi, S Sakshi, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha
**Year:** June 2024 (EMNLP 2024)
**ArXiv:** [2406.11768](https://arxiv.org/abs/2406.11768)
**Affiliation:** University of Maryland

### Architecture
- **Base LLM:** LLaMA-2-7B-chat (~7B parameters total)
- **Audio encoders:** Audio Spectrogram Transformer (AST, ~85M params, fine-tuned on AudioSet) + custom Audio Q-Former (~280M params, BERT-based, trained on 2.5M+ audio-caption pairs)
- **Multi-layer aggregator:** Takes features from multiple AST layers (not just final layer), unlike prior work
- **Fine-tuning:** LoRA modules adding ~4.2M learnable parameters
- **Training data:** OpenAQA + AudioSet-Strong (102K), AudioSet (500K), VGGSound, FSD50K, AudioCaps, Clotho, NSynth, MusicCaps, MusicQA — ~1.67M samples, 6.3M QA pairs
- **CompA-R dataset:** 62,613 AudioSet-Strong samples → 200,234 complex reasoning instruction-response pairs

### Supported Tasks
Non-speech audio classification (zero-shot, in-domain), audio captioning, dense captioning, closed-ended QA, open-ended QA, complex audio reasoning, hallucination evaluation

### Target Speaker Extraction
**No.** GAMA is an audio understanding/reasoning model; outputs text only. No separation.

### Open Source
**Yes (partial).** Code on GitHub: [Sreyan88/GAMA](https://github.com/Sreyan88/GAMA). Project page: [sreyan88.github.io/gamaaudio](https://sreyan88.github.io/gamaaudio/).

---

## PART 8: AudioPaLM (Google)

**Full Name:** AudioPaLM: A Large Language Model That Can Speak and Listen
**Authors:** Paul K. Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zalán Borsos, Félix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, Hannah Muckenhirn, Dirk Padfield, James Qin, Danny Rozenberg, Tara Sainath, Johan Schalkwyk, Matt Sharifi, Michelle Tadmor Ramanovich, Marco Tagliasacchi, Alexandru Tudor, Mihajlo Velimirović, Damien Vincent, Jiahui Yu, Yongqiang Wang, Vicky Zayats, Neil Zeghidour, Yu Zhang, Zhishuai Zhang, Lukas Zilka, Christian Frank
**Year:** June 2023
**ArXiv:** [2306.12925](https://arxiv.org/abs/2306.12925)

### Architecture
- Merges **PaLM-2** (text LLM) with **AudioLM** (speech/audio token LM) into a single unified vocabulary multimodal model
- Extends PaLM-2's text vocabulary with audio tokens from SoundStream neural codec
- Operates on interleaved text + audio token sequences for end-to-end speech understanding and generation
- Model size: not publicly disclosed (PaLM-2 ranges from 8B–340B; specific AudioPaLM configuration unpublished)

### Supported Tasks
Automatic speech recognition (ASR), automatic speech translation (AST), speech-to-speech translation (S2ST) with voice preservation, zero-shot translation to unseen language pairs, voice transfer across languages

### Target Speaker Extraction
**No.** AudioPaLM is designed for ASR/translation with voice preservation; it does not perform source separation.

### Open Source
**No.** Research paper and demo examples only at [google-research.github.io/seanet/audiopalm/examples](https://google-research.github.io/seanet/audiopalm/examples/). No weights or code released.

---

## PART 9: Gemini Audio (Google DeepMind)

**Full Name:** Gemini 2.5 Native Audio / Gemini Audio
**Year:** 2024–2025 (ongoing development; 2.5 native audio capabilities announced 2025)

### Architecture
Gemini is a natively multimodal model built end-to-end to process text, images, audio, video, and code. Audio is processed natively (not as a downstream add-on). Internal architecture undisclosed; weights not released. Supports up to 9.5 hours of audio in a single context window.

### Supported Audio Tasks
- Transcription and multilingual ASR
- Speech-to-speech translation (70+ languages, 2,000+ language pairs)
- Speaker diarization (distinguishing and labeling multiple speakers)
- Emotion, intent, and non-verbal cue detection (laughter, sighs, whispers)
- Noise reduction / voice activity detection (preprocessing pipeline)
- Generative text-to-speech (Gemini 2.5 TTS, studio quality)
- Real-time voice agents (Gemini Live)

### Target Speaker Extraction / Separation
**Limited, implicit.** Gemini can perform speaker diarization (label who spoke when) and can "disregard background speech and ambient conversations." This is not source separation in the classical waveform-domain sense — it does not output a clean audio file of one speaker. It labels and transcribes. A speech AI practitioner would not use Gemini as a replacement for SepFormer, SpEx+, or SAM Audio.

### Open Source
**No.** Closed commercial API only. Available via Gemini API / Google AI Studio / Vertex AI.

---

## PART 10: AudioBox (Meta FAIR)

**Full Name:** Audiobox: Unified Audio Generation with Natural Language Prompts
**Authors:** Apoorv Vyas, Bowen Shi, Matthew Le, Andros Tjandra, Yi-Chiao Wu, and 19 others (Meta FAIR)
**Year:** December 2023
**ArXiv:** [2312.15821](https://arxiv.org/abs/2312.15821)

### Architecture
- **Flow-matching based** unified generative model (related to Meta's Voicebox)
- Supports description-based and example-based prompting
- Independent control of transcript, vocal style, and acoustic style
- Self-supervised pre-training with infilling objective on unlabeled audio
- Uses Bespoke Solvers for >25x generation speedup over default ODE solver
- Reports 0.745 similarity on LibriSpeech (zero-shot TTS), 0.77 FAD on AudioCaps (text-to-sound)

### Key Relation to Separation
The paper notes the pre-trained model "demonstrates promising improvements on other speech generation tasks, including source separation and speech enhancement" — AudioBox's flow-matching pre-training creates representations that transfer to separation tasks. **SAM Audio (2025) from the same Meta FAIR group builds directly on AudioBox's flow-matching architecture.**

### Open Source
**Partial.** Demo at [audiobox.metademolab.com](https://audiobox.metademolab.com). Full weights not open sourced; Audiobox Aesthetics (quality assessment spinoff) was open-sourced separately.

---

## PART 11: VoiceCraft

**Full Name:** VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild
**Authors:** Puyuan Peng, Po-Yao Huang, Shang-Wen Li, Abdelrahman Mohamed, David Harwath
**Year:** March 2024 (ACL 2024)
**ArXiv:** [2403.16973](https://arxiv.org/abs/2403.16973)

### Architecture
- **Token infilling neural codec language model**
- Transformer decoder with a novel **token rearrangement procedure**: causal masking (masked spans moved to end of sequence) + delayed stacking (codebook tokens shifted by index in time dimension)
- Enables generation *within* an existing sequence — i.e., fill in a gap, not just append
- Two model sizes: **330M** and **830M** parameters
- Training data: GigaSpeech (9,000 hours of YouTube/podcast audio)
- Codec: EnCodec-style neural codec (multiple codebooks)

### Supported Tasks
Speech editing (change specific words/phrases in existing recordings), zero-shot TTS (clone voice from short reference), voice continuation

### Target Speaker Extraction
**No.** VoiceCraft generates/edits speech; it does not separate speakers from mixtures. It is relevant to the downstream use case of *re-synthesizing* a target speaker's voice once you have reference audio, but is not a separation model.

### Open Source
**Yes.** [github.com/jasonppy/VoiceCraft](https://github.com/jasonppy/VoiceCraft), weights on Hugging Face (`giga330M`, `giga830M`). License: CC BY-SA 4.0.

---

## PART 12: WavJourney

**Full Name:** WavJourney: Compositional Audio Creation with Large Language Models
**Authors:** Xubo Liu, Zhongkai Zhu, Haohe Liu, Yi Yuan, Qiushi Huang, Jinhua Liang, Yin Cao, Qiuqiang Kong, Mark D. Plumbley, Wenwu Wang
**Year:** July 2023
**ArXiv:** [2307.14335](https://arxiv.org/abs/2307.14335)
**Affiliation:** Audio-AGI group / University of Surrey

### Architecture
**Not a single neural model.** WavJourney is an LLM-orchestration framework:
1. LLM (GPT-4 or similar) receives a text description and writes a structured audio script
2. Script compiler converts the script to executable Python-like code
3. Code calls task-specific expert models (TTS, text-to-music, text-to-SFX, mixing tools)
4. Output: composed multi-element audio file

No training of new models; no fine-tuning of LLMs; leverages existing specialist models as tools.

### Target Speaker Extraction
**No.** WavJourney is a generation framework. No separation capability.

### Open Source
**Yes.** [github.com/Audio-AGI/WavJourney](https://github.com/Audio-AGI/WavJourney)

---

## PART 13: SALMONN / Whisper Separation Connection

**Whisper as a Separation Aid (not a separation model):**
OpenAI Whisper (arXiv 2212.04356) is an ASR encoder-decoder model trained on 680,000 hours of weakly supervised multilingual data. It has no separation capability built-in. However, it is being used as a *foundation* for separation-adjacent work:

1. **WhisperX** (INTERSPEECH 2023): Combines Whisper with pyannote for word-level timestamps + diarization
2. **TSELM** (arXiv 2409.07841, 2024): Uses WavLM discrete tokens + cross-attention for **target speaker extraction** — uses language model (encoder-only) to generate reconstructed tokens, then HiFi-GAN for resynthesis. Achieves state-of-the-art speech quality in TSE.
3. **Whisper-TSE joint optimization** (arXiv 2501.14477, 2025): Jointly optimizes Whisper ASR with a target speech extraction network, using Whisper's encoder as the shared speech representation backbone
4. **SoundBeam + M2D** (arXiv 2409.12528, 2024): Uses M2D (Masked-Modeling Duo) audio foundation model features inside the SoundBeam TSE framework; shows foundation model features improve extraction, especially with enrollment clues

---

## PART 14: UniAudio 1.5

**Full Name:** UniAudio 1.5: Large Language Model-driven Audio Codec is A Few-shot Audio Task Learner
**ArXiv:** [2406.10056](https://arxiv.org/abs/2406.10056)
**Year:** June 2024 (NeurIPS 2024)

### Architecture
- **LLM-Codec:** Semantic-guided multi-scale RVQ codec that represents audio tokens as words/sub-words in the LLM's vocabulary
  - Layer 1: semantic information
  - Layer 2: coarse-grained acoustics
  - Layer 3: residual acoustics
- Compresses 1 second of audio to **57 tokens** (extremely compact vs. prior codecs)
- Frozen LLM performs **cross-modal in-context learning** over audio token sequences — no parameter updates required for new tasks
- Codec trained on 2,000 hours of audio

### Supported Tasks (few-shot)
Speech emotion classification, audio classification, TTS, speech enhancement — via in-context examples only

### Target Speaker Extraction
Not demonstrated in the paper, but the architecture supports it through in-context examples.

---

## PART 15: Summary Comparison Table

| Model | Year | Organization | Params | Does Separation? | TSE (Speaker)? | Open Source? | ArXiv |
|---|---|---|---|---|---|---|---|
| **SAM Audio** | 2025 | Meta FAIR | 500M–3B | **Yes — primary task** | Text/visual attr. only (no enrollment) | Yes (SAM License) | 2512.18099 |
| **UniAudio** | 2023 | CUHK/MSRA | ~1B | Yes (speech extraction task) | **Yes (audio enrollment)** | Yes (MIT) | 2310.00704 |
| **SALMONN** | 2023 | Tsinghua/ByteDance | ~13B | No (text output only) | No | Yes | 2310.13289 |
| **Qwen2-Audio** | 2024 | Alibaba | 8.2B | No (text output only) | No | Yes (Apache) | 2407.10759 |
| **GAMA** | 2024 | UMD | ~7B | No (text output only) | No | Partial | 2406.11768 |
| **AudioPaLM** | 2023 | Google | Undisclosed | No | No | No | 2306.12925 |
| **Gemini Audio** | 2024–25 | Google DeepMind | Undisclosed | Diarization only (no waveform sep) | No | No | — |
| **AudioBox** | 2023 | Meta FAIR | Undisclosed | Transfer learning benefit | No | Partial | 2312.15821 |
| **Stable Audio** | 2024 | Stability AI | ~1.21B | No (generation only) | No | Yes (CC) | 2407.14358 |
| **VoiceCraft** | 2024 | UT Austin/Meta | 330M/830M | No (editing/TTS) | No | Yes (CC BY-SA) | 2403.16973 |
| **WavJourney** | 2023 | Audio-AGI | Framework | No (generation framework) | No | Yes | 2307.14335 |
| **TSELM** | 2024 | Duke | ~100M range | TSE only | **Yes (WavLM tokens)** | Yes | 2409.07841 |
| **SoundBeam+M2D** | 2024 | NTT | — | TSE with FM features | **Yes (enrollment)** | — | 2409.12528 |

---

## PART 16: Key Findings for Speech Separation Use Cases

### Models That Can Actually Separate Audio Waveforms
1. **SAM Audio** (Meta, 2025): Best for general separation, text/visual/temporal prompts; no speaker enrollment
2. **UniAudio** (2023): Explicit TSE task with 3-second speaker audio enrollment; 1B params; open source
3. **TSELM** (2024): Purpose-built TSE using discrete WavLM tokens + language model; enrollment-based; open source

### Models Useful as Feature Extractors / Transfer Learning for Separation
- **SoundBeam + M2D** (2024): Shows that audio foundation model features (M2D) improve TSE systems
- **Whisper encoder** (2022): Used as backbone in joint ASR+TSE systems
- **AudioBox** pre-training (2023): Flow-matching representations transfer to separation

### Models That Only Output Text (Cannot Produce Separated Audio)
SALMONN, Qwen2-Audio, GAMA, AudioPaLM, Gemini Audio — these are all audio-language understanding models. They can *describe* or *transcribe* audio, including multi-speaker audio, but cannot produce separated waveform outputs.

### Models for Generation (Not Separation)
Stable Audio, VoiceCraft, WavJourney, AudioBox — these generate audio but do not perform separation.

---

## Sources

- [SAM Audio arXiv 2512.18099](https://arxiv.org/abs/2512.18099)
- [SAM Audio Meta Blog](https://ai.meta.com/blog/sam-audio/)
- [SAM Audio GitHub](https://github.com/facebookresearch/sam-audio)
- [UniAudio arXiv 2310.00704](https://arxiv.org/abs/2310.00704)
- [UniAudio 1.5 arXiv 2406.10056](https://arxiv.org/abs/2406.10056)
- [SALMONN arXiv 2310.13289](https://arxiv.org/abs/2310.13289)
- [Qwen2-Audio arXiv 2407.10759](https://arxiv.org/abs/2407.10759)
- [GAMA arXiv 2406.11768](https://arxiv.org/abs/2406.11768)
- [AudioPaLM arXiv 2306.12925](https://arxiv.org/abs/2306.12925)
- [AudioBox arXiv 2312.15821](https://arxiv.org/abs/2312.15821)
- [Stable Audio Open arXiv 2407.14358](https://arxiv.org/abs/2407.14358)
- [VoiceCraft arXiv 2403.16973](https://arxiv.org/abs/2403.16973)
- [WavJourney arXiv 2307.14335](https://arxiv.org/abs/2307.14335)
- [TSELM arXiv 2409.07841](https://arxiv.org/abs/2409.07841)
- [SoundBeam+M2D arXiv 2409.12528](https://arxiv.org/abs/2409.12528)
- [Gemini Audio DeepMind](https://deepmind.google/models/gemini-audio/)
- [SALMONN GitHub](https://github.com/bytedance/SALMONN)
- [Qwen2-Audio GitHub](https://github.com/QwenLM/Qwen2-Audio)