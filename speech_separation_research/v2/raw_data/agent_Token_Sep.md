I now have sufficient data for a thorough and authoritative report. Let me write it.

---

# Deep Research Report: Token-Based and Codec-LLM Approaches to Speech Separation and Target Speaker Extraction

**Date:** April 2026 | **Purpose:** Production system redesign — no compute constraints, best possible approach.

---

## Executive Summary

The period 2023–2026 has produced a genuine paradigm shift in speech separation and target speaker extraction (TSE). Discriminative mask-based systems (SpEx+, TF-GridNet, DPCCN) remain the gold standard on objective signal metrics such as SI-SDR but are being systematically beaten on perceptual quality, naturalness, speaker similarity, and — critically — out-of-domain generalization by generative approaches. The evidence is now strong enough to conclude: **token-based and generative methods do not yet beat discriminative methods on SI-SDR in controlled benchmarks, but they achieve superior perceptual quality, speaker fidelity, and robustness on real-world audio, and that gap on signal metrics is rapidly closing.** For a production system where the final listener is a human (or a downstream ASR/speaker-ID system), the optimal architecture as of early 2026 is a **discriminative front-end + generative back-end hybrid**, with pure flow-matching generative pipelines as a close and rapidly improving alternative.

---

## Part 1: Audio Codec Tokenization Foundations

### 1.1 SoundStream (Google, 2021)
The foundational paper establishing the neural audio codec paradigm. SoundStream introduced the encoder–residual vector quantizer (RVQ)–decoder architecture that all subsequent codecs inherited. The core idea: an encoder maps raw waveform to a continuous embedding, RVQ discretizes it into multiple streams of tokens (one per codebook), and a decoder reconstructs the waveform. At 3 kbps with 8 quantizers, SoundStream produced speech indistinguishable from the original in listening tests. For separation research, SoundStream matters because it established the token representation used by UniSep and early codec separation experiments.

### 1.2 EnCodec (Meta, arXiv 2210.13438, 2022)
**Architecture:** Convolutional encoder → RVQ (8 codebooks, vocabulary 1024 each) → convolutional decoder. At 24 kHz, 6 kbps produces 75 Hz token streams (75 frames/sec × 8 codebooks). At lower rates, tokens are generated at 50 or 25 Hz.

**Why it matters for separation:** EnCodec became the default codec for codec-language-model research (VALL-E, SpeechX, SLM-SS) because its code was released and its compression rate was tractable for LM training. For TSE, the standard formulation is: given mixed-speech tokens `C(s₁+s₂)` and enrollment tokens `C(sᵣₑf)`, predict clean target tokens `C(s₁)`. This is a sequence-to-sequence classification problem, which can be handled by any LM-style architecture.

**Key limitation for separation:** EnCodec was trained on solo speech/music and has no exposure to mixtures. Its internal representation is therefore not specialized for disentangling overlapping sources. The RVQ structure means that coarse codebooks (layer 1) capture primarily semantic/prosodic content and fine codebooks (layers 6–8) capture acoustic detail. Separation errors in fine codebooks produce harmless frequency smearing; errors in coarse codebooks (hallucinating the wrong phoneme or speaker) are catastrophic.

**Code:** https://github.com/facebookresearch/encodec

### 1.3 DAC — Descript Audio Codec (arXiv 2306.06546, 2023)
**Improvement over EnCodec:** DAC introduces improved adversarial training losses, Snake activations, multi-band discriminators, and periodic activation functions. It handles 44.1 kHz audio with 9 RVQ codebooks at 8 kbps with significantly better high-frequency reconstruction and reduced perceptual artifacts compared to EnCodec.

**For separation:** DAC became the preferred codec for separation in the codec-embedding-space approach (Codecformer, Codecformer-EL, AnyEnhance). The codec's encoder embedding space is 1024-dimensional; separation can be performed directly in this space without ever running the decoder during training, saving 97x multiply-accumulate operations.

**Key property:** DAC operates at 86 tokens/sec at 44.1 kHz, producing far fewer tokens than waveform methods, which makes transformer attention tractable on longer sequences.

**Code:** https://github.com/descriptinc/descript-audio-codec

### 1.4 SpeechTokenizer (arXiv 2308.16692, ICLR 2024)
SpeechTokenizer makes the most important architectural advance for downstream separation work: **semantic–acoustic disentanglement**. By distilling a HuBERT/WavLM model into RVQ layer 1 via a teacher loss, SpeechTokenizer forces layer 1 to capture phonetic/semantic content (like discrete HuBERT) while layers 2–8 capture only speaker identity, prosody, and fine acoustics.

**Why this matters for TSE:** When a generative model must predict clean speaker tokens from a mixture, if RVQ layer 1 captures "what was said" and layers 2+ capture "who said it," you can train a semantic LM on layer 1 (a tractable, low-vocabulary problem) and a separate acoustic model for layers 2+. This is exactly the two-stage design used by GenTSE and Metis.

**Code:** https://github.com/ZhangXInFD/SpeechTokenizer

### 1.5 How RVQ Tokens Enable the Separation Problem
The "given mixed tokens, predict clean tokens" framing has several important properties:

- The problem is a **conditional sequence-to-sequence classification**, which maps directly to encoder-decoder or decoder-only transformer architectures
- Because token prediction is a categorical distribution, models naturally express uncertainty and can use beam search, sampling, or temperature control
- The codec decoder guarantees natural-sounding output even when the predicted tokens are a soft compromise — generative artifacts replace the harsh over-subtraction artifacts of mask-based methods
- **Critical failure mode:** autoregressive LM token prediction can **hallucinate** — generating plausible-sounding tokens that correspond to words not present in the mixture. This is the primary failure mode of purely generative codec approaches, and the reason all competitive 2025 systems use semantic tokens separately from acoustic tokens or add a discriminative front-end.

---

## Part 2: Token-Based Separation Papers — Detailed Analysis

### 2.1 Codecformer / "Towards Audio Codec-based Speech Separation"
**arXiv:** 2406.12434 | **Venue:** INTERSPEECH 2024 | **Authors:** Yip, Zhao, Ng, Chng, Ma (Nanyang / ByteDance)

**Approach:** Discriminative separation *inside* the DAC embedding space (continuous, pre-quantization). A 16-layer Transformer with 256-dim embeddings operates on DAC encoder outputs at compressed temporal resolution, producing masked-and-scaled embeddings that are passed to the DAC decoder for reconstruction. No token prediction — this is mask-based but in codec-latent space.

**Metrics (WSJ0-2mix):**
- Codec SI-SDRi (cSI-SDRi): 9.9 dB (vs. 9.6 dB conventional Sepformer baseline using same codec-distorted comparison)
- PESQ: 2.58 (slightly below clean-reference Sepformer at 2.75)

**Computational advantage:** 52x reduction in multiply-accumulate operations vs. waveform-domain Sepformer; 2.7x faster training.

**Training data:** WSJ0-2mix (30 h training). Single GPU (V100).

**Model size:** ~16M parameters (16 Transformer blocks, 256-dim).

**Failure modes:** DAC trained only on solo speech; its embedding space lacks representations for fully-overlapping speech, degrading separation quality on dense overlap. RVQ quantization itself introduces 2–3 dB cSI-SDRi degradation vs. oracle.

**Code:** https://github.com/Yip-Jia-Qi/codecformer

---

### 2.2 Codecformer-EL — "Speech Separation using Neural Audio Codecs with Embedding Loss"
**arXiv:** 2411.17998 | **Date:** November 2024 | **Authors:** Yip, Kwok, Ma, Chng

**Approach:** Evolution of Codecformer. Proposes an *embedding loss* that trains the separation model by comparing encoder representations directly (no waveform decoding needed at training time). This means only the codec encoder is needed during training, not the decoder.

**Metrics (WSJ0-2mix):** Comparative improvements over Codecformer are 2x training speed, 1.9x fewer MACs. Objective scores (SI-SDR) slightly lower than Codecformer-waveform, but DNSMOS and perceptual quality are superior.

**Key insight:** The embedding space is perceptually smoother than the waveform. Optimizing the loss in embedding space inadvertently optimizes for perceptual quality at the cost of signal-fidelity metrics. This prefigures the broader finding: **codec-domain training aligns with perceptual quality, not SI-SDR**.

**Failure modes:** Constrained by codec quality ceiling; codec distortions are penalized by traditional objective metrics but are perceptually transparent.

---

### 2.3 SpeechX (Microsoft, arXiv 2308.06873)
**Venue:** IEEE/ACM TASLP 2024 | **Authors:** Wang, Thakker, Chen et al. (Microsoft)

**Approach:** Multi-task neural codec language model using EnCodec (8 codebooks, 75 Hz). Tasks include zero-shot TTS, noise suppression, TSE, speech removal, and speech editing. TSE is framed autoregressively: input sequence = `[enrollment_tokens, <TSE_task_token>, mixture_tokens]`; output = `target_speaker_tokens`.

**Architecture:** 12-layer Transformer, 16 heads, 1024-dim embeddings, with separate autoregressive (AR) stage (coarse codes) and non-autoregressive (NAR) stage (fine codes), analogous to VALL-E.

**Training data:** 60,000 hours (LibriLight + simulated mixtures at SIR −5 to +20 dB).

**Model size:** Not explicitly stated but consistent with VALL-E scale (~300M parameters).

**TSE Metrics (LibriSpeech test-clean):**
- WER: 2.53% (with text prompt) / 5.00% (audio-only)
- DNSMOS: 3.46 / 3.01
- PESQ: 2.28 / 2.23
- Speaker Similarity: ~0.58

**Comparison to baseline:** Substantially outperforms VoiceFilter (WER 5.09%) on intelligibility. DNSMOS comparable.

**Critical observation:** Speaker similarity of 0.58 is relatively low, reflecting that the autoregressive codec model does not always faithfully reproduce the acoustic identity of the target. This is the hallucination problem manifesting as speaker drift.

**Code:** Demo only at https://aka.ms/speechx; model weights not released.

---

### 2.4 TSELM — "Target Speaker Extraction using Discrete Tokens and Language Models"
**arXiv:** 2409.07841 | **Date:** September 2024 | **Authors:** Tang, Zeng, Li (Duke Kunshan University)

**Approach:** Converts TSE from regression to classification. Uses 6 layers of WavLM Large (layers 1, 3, 7, 12, 18, 23), applies K-means clustering (K=1000) per layer to produce discrete token streams. A cross-entropy-trained encoder-only Transformer with cross-attention over enrollment embeddings predicts the clean target's token sequence, which is then decoded by a scalable HiFi-GAN vocoder.

**Architecture:** Three scales (S/M/L). TSELM-L: 768-dim, 12 layers, 16 heads.

**Training data:** LibriSpeech train-clean-[100+360] (~500 h). Synthetic mixtures, 0–5 dB relative SNR. 3-second mixtures, 4-second reference.

**Metrics (Libri2Mix test):**
- DNSMOS OVRL: 3.49 (competitive with generative peers)
- dWER (differential word error rate): 27.5 (higher/worse than SpEx+ at comparable settings — intelligibility is a weakness)
- Speaker Similarity: 0.895 (against discretized reference; lower vs. continuous reference)

**Key findings:** DNSMOS (perceptual quality) is excellent; the discrete representation preserves naturalness. But intelligibility (dWER) suffers because K-means discretization on clean speech may emphasize dominant-speaker tokens in a mixture.

**Failure modes:** K-means trained on solo clean speech is mismatched to mixture inputs; information loss from discretization reduces speaker similarity; concatenation strategy for reference audio is essential (without it, performance degrades sharply).

**Code:** Available (GitHub repository referenced in paper).

---

### 2.5 LauraTSE — "Target Speaker Extraction using Auto-Regressive Decoder-Only Language Models"
**arXiv:** 2504.07402 | **Date:** April 2025 | **Authors:** Tang, Zeng, Li (Duke Kunshan University)

**Approach:** First single-task TSE model using a decoder-only autoregressive LM (LauraGPT backbone). Takes continuous log-mel spectrograms (not discrete tokens) of mixture and reference as input. Predicts the first n RVQ layers of FunCodec for coarse representation, then an encoder-only vocoder model completes fine-grained detail via one-step prediction.

**Architecture:** AR decoder: 10 Transformer blocks, 8 heads, 512-dim (36M params). Total: ~77M parameters.

**Training data:** 460 h LibriSpeech clean. 100 epochs on 16 GPUs.

**Metrics (Libri2Mix test, compared to competing methods):**

| Model | Type | DNSMOS OVRL | dWER ↓ | WavLM Sim |
|---|---|---|---|---|
| SpEx+ | Discriminative | 3.000 | 3.029 | 0.840 |
| WeSep | Discriminative | 3.228 | 4.041 | 0.922 |
| TSELM-L | Generative (tokens) | 3.228 | 4.029 | 0.808 |
| AnyEnhance | Generative (masked) | 3.353 | 4.277 | 0.735 |
| **LauraTSE** | **Generative (AR)** | **3.336** | **4.333** | **0.908** |

**Key finding:** LauraTSE's continuous input representation (log-mel, not discrete tokens) is critical. Replacing continuous input with WavLM discrete tokens significantly reduces speaker similarity. Using continuous input, the model achieves 0.908 WavLM speaker similarity — better than TSELM (0.808) and AnyEnhance (0.735), and close to the discriminative WeSep (0.922).

**Failure modes:** Higher dWER than SpEx+ (autoregressive drift causing occasional word hallucinations); pretrained encoder caused infinite looping in some ablation configurations; performance degrades when only discrete tokens used as input.

**Code:** Demo website available; code referenced.

---

### 2.6 GenTSE — "Enhancing Target Speaker Extraction via a Coarse-to-Fine Generative Language Model"
**arXiv:** 2512.20978 | **Date:** December 2024 | **Authors:** Not fully listed in fetched content

**Approach:** Two-stage decoder-only LM. Stage 1 predicts *semantic tokens* (WavLM features + K-means discretization) from WavLM continuous embeddings. Stage 2 predicts *acoustic tokens* (SimCodec, a single-codebook neural audio codec). Both stages use continuous SSL/codec embeddings as input.

**Key innovations:**
- **Frozen-LM Conditioning (FLC):** Trains Stage 2 on tokens predicted by an earlier frozen Stage 1 checkpoint rather than ground-truth semantic tokens. This reduces the training-inference mismatch that causes hallucination in autoregressive models.
- **Direct Preference Optimization (DPO):** Uses a proxy MOS scorer (UTMOS) to generate preference pairs and fine-tunes the acoustic stage to prefer higher-quality outputs.

**Architecture:** 12 layers, 8 heads, 1024 hidden dim. Training on LibriMix 2-speaker clean (train-100 + train-360).

**Metrics (Libri2Mix clean):** GenTSE outperforms TSELM-L, LLaSE-G1, and Metis across DNSMOS, NISQA, UTMOS, and SECS (speaker embedding cosine similarity). Discriminative USEF-SepFormer achieves lower dWER (2.880 vs. 3.976 for GenTSE), confirming generative methods still lag on intelligibility.

**Failure modes:** DPO tuning for quality causes minor speaker similarity degradation when not combined with cross-entropy loss; removing the semantic stage substantially increases dWER; the system is not publicly released.

**Code:** Not released at time of research.

---

### 2.7 SLM-SS — "Speech Language Model for Generative Speech Separation"
**arXiv:** 2601.19533 | **Date:** January 2026 | **Authors:** Not specified

**Approach:** Encoder-decoder model framing speech separation as multi-codebook sequence generation. WavLM-large as encoder; Whisper-medium dimensions for the 16-layer decoder. Separates two speakers by predicting EnCodec tokens (8 codebooks) — autoregressive for coarse codebooks, non-autoregressive for fine codebooks.

**Architecture:** ~600M parameters total.

**Training data:** LibriMix (100 h + 360 h).

**Metrics (Libri2Mix):**
- WER: 7.24%
- Speaker Similarity: 91.7%
- LPS (Levenshtein Phoneme Similarity): 0.954
- SpeechBERTScore: 0.913
- MOS: 4.19

**Key finding:** Outperforms BSRNN and Sepformer on linguistic consistency (LPS, SpeechBERT) despite comparable subjective quality. MOS of 4.19 is notably high. WER of 7.24% is worse than clean discriminative methods, consistent with the pattern.

---

### 2.8 CodeSep — "Low-Bitrate Codec-Driven Speech Separation"
**arXiv:** 2601.12757 | **Date:** January 2026 | **Authors:** Not listed

**Approach:** Joint separation-and-compression. A Base-Token Disentanglement (BTD) module uses anti-consistency source-inter Transformers to produce separated base tokens (1 codebook per speaker); an Auxiliary-Token Serial Prediction (ATSP) module predicts remaining codec layers client-side from 1 kbps base stream.

**Key use case:** This is the only paper explicitly designed for *low-bitrate transmission* of separated speech — separation and compression are performed simultaneously at 1 kbps per speaker.

**Training data:** LibriMix-clean 16 kHz (270 h train).

**Metrics:**
- UTMOS: 3.14, DNSMOS: 3.67, NMOS: 3.65, SMOS (speaker similarity): 3.43
- Deliberately avoids SI-SDR, arguing it is unsuitable for codec-based methods

**Code:** Demo only at https://redmist328.github.io/CodeSep/

---

### 2.9 SepALM — "Audio Language Models Are Error Correctors for Robust Speech Separation"
**arXiv:** 2505.03273 | **Venue:** IJCAI 2025 | **Authors:** Mu, Yang, Wang

**Approach:** Hybrid cascade. SepFormer (26M) performs initial discriminative separation. SpeechGPT-7B (with CoT prompting and LoRA fine-tuning) corrects the transcript in text space. A DAC-based codec language model (202M) resynthesizes the corrected speech. A 2-layer CNN aligner performs phase compensation against the original mixture.

**Training data:** WHAM!, WHAMR!, Libri2Mix combined.

**Metrics:**
- SI-SNRi: **17.6 dB** (Libri2Mix) — highest reported in any token/codec-based system
- SDRi: 18.2 dB
- NMOS: 3.91
- WER: 3.76–4.03%
- Out-of-domain (MUSAN/DEMAND): 13.9–14.4 dB SI-SNRi

**Key insight:** By correcting errors in text space (where noise is absent), then resynthesizing, SepALM achieves +4.5 dB SI-SNRi improvement over the SepFormer discriminative baseline alone. This is the highest SI-SNRi reported by any codec/LLM system, achieved by using the LLM as a "corrector" rather than a primary separator.

**Why this matters:** SepALM demonstrates that the optimal way to use LLMs for separation in 2025 is as a *post-processing corrector* on top of a discriminative separator, not as the primary separation mechanism.

---

### 2.10 UniSep — "Universal Target Audio Separation with Language Models at Scale"
**arXiv:** 2503.23762 | **Date:** March 2025 | **Authors:** Wang et al. (CUHK / Tencent / Tsinghua)

**Approach:** Causal decoder-only LM (535M parameters) operating in SoundStream RVQ token space (3 codebooks). Separates arbitrary audio across speech, music, and sound domains from a single model. Input sequence: `[mixture_tokens, prompt_tokens, target_tokens]`.

**Pre-training innovation:** Two self-supervised tasks on audio-only (no mixture labels): audio continuation and audio inpainting (20% token masking). This reduces reliance on expensive mixture simulation.

**Training data:** 36.5k hours total (20k LibriLight pre-train + 5.8k AudioSet + 10k+ supervised speech/sound/music).

**Architecture:** 12 global + 4 local transformer layers, 8 heads, 1536-dim.

**Metrics (Libri2Mix speech):**
- PESQ: 2.23
- DNSMOS: 4.07
- MUSHRA: 4.11±0.11

**Key differentiator:** Universal model; single model for all audio domains. Foundation model capability demonstrated by fine-tuning with only 16.7 hours to achieve 96.94% of ViSQOL vs. AudioSep trained on 14,000 hours. Represents the codec-LM approach scaled to foundation model size.

**Code:** Demo at https://uniseparation.github.io/UniSep/ — full code not released.

---

## Part 3: LLM-Guided Speech Separation

### 3.1 SpeechVerse (Amazon, 2024)
Multi-task speech LLM (Whisper + LLaMA) trained with supervised instruction fine-tuning across 11 speech tasks via natural language prompts. Outperforms conventional baselines on 9 of 11 tasks. **Does not include speaker separation or TSE as primary tasks**; primarily targets ASR, ST, speaker identification, intent classification. Not a competitive separation model.

### 3.2 SALMONN (Tsinghua / ByteDance, arXiv 2310.13289, ICLR 2024)
Window-level Q-Former fusing Whisper (speech) and BEATs (audio events) encoders into a Vicuna-7B LLM. Enables audio-speech co-reasoning. **Does not perform speech separation or TSE** as a trained task. The paper demonstrates music analysis, acoustic scene classification, and multilingual ASR. Separation capability would require significant fine-tuning on paired mixture data, which has not been reported.

### 3.3 Qwen-Audio / Qwen2-Audio (Alibaba, 2023–2024)
Qwen-7B extended to audio via an audio encoder. Instruction-following across 30+ audio tasks. **TSE is not among the reported tasks.** Like SALMONN, it is a general audio-language model that could theoretically be instruction-fine-tuned for TSE but no published work does so.

### 3.4 AudioPaLM (Google, arXiv 2306.12925)
Fuses PaLM-2 (text LLM) with AudioLM (audio codec LM). Processes speech as SoundStream tokens. Tasks: ASR, speech-to-speech translation, voice cloning. **Does not perform speech separation.** AudioPaLM's value for this research is demonstrating that text LLMs can guide audio token generation with strong linguistic coherence — the architecture is adaptable to TSE but has not been applied.

### 3.5 AudioLM (Google, arXiv 2209.03143)
Hierarchical framework: SoundStream (coarse acoustic tokens) + w2v-BERT (semantic tokens). Generates speech/music continuations. **Explicitly not a separation model.** Relevant as the architectural ancestor of SpeechX and GenTSE's two-stage approach.

### 3.6 SoundStorm (Google, arXiv 2305.09636)
Non-autoregressive parallel audio codec token generation using bidirectional attention and confidence-based parallel decoding. **Not designed for separation.** Relevant as a faster alternative to autoregressive codec decoding — two orders of magnitude faster than AudioLM's sequential generation.

### 3.7 TWIST (arXiv 2305.13009)
"Textually Pretrained Speech Language Models" — warm-starts speech LM training from a text LLM (OPT/LLaMA). Produces 7B and 13B parameter speech LMs. **Does not perform separation.** Relevant as methodology: initializing a speech LM from a text LLM dramatically accelerates convergence and improves linguistic coherence of generated tokens. This principle is directly applicable to codec-based TSE models.

### 3.8 LLaSE-G1 — "Incentivizing Generalization Capability for LLaMA-based Speech Enhancement"
**arXiv:** 2503.00493 | **Venue:** ACL 2025 | **Authors:** Kang, Zhu, Zhang et al. (Northwestern Polytechnical / HKUST / Huawei)

**Approach:** 16-layer LLaMA architecture (~1.07B parameters) with WavLM continuous input representations and X-Codec2 (semantic+acoustic) output tokens. Trained on ~5,000 h across 5 tasks: noise suppression, packet loss concealment, echo cancellation, speech separation, **and TSE**.

**Key capability:** Demonstrates emergent zero-shot speech separation on tasks not explicitly in training — the LLM generalizes to unseen enhancement tasks. This is the clearest demonstration that large-scale LLM-based speech models develop generalizable separation capabilities.

**Metrics:** State-of-the-art across DNSMOS, AECMOS, PLCMOS, and speaker similarity on DNS Challenge and reverberant benchmarks. Competitive with specialized discriminative models.

**Code:** Released (GitHub repository).

### 3.9 WavLLM (arXiv ~2406.12428)
WavLM used primarily as a speech encoder feeding into LLMs for understanding tasks (ASR, emotion, speaker recognition). **No speech separation task demonstrated.** WavLM features are used as *input* to LLMs, not for generative separation.

---

## Part 4: Generative TSE — Diffusion, Flow Matching, and Hybrid Methods

This is the most active frontier in 2024–2026, representing the most important finding for production system design.

### 4.1 DDTSE — "Discriminative Diffusion Model for Target Speech Extraction"
**arXiv:** 2309.13874 | **IEEE ICASSP 2025**

**Approach:** Combines diffusion model's forward process with discriminative model's reconstruction loss. Two-stage training: first trains a discriminative mask predictor, then uses its output to guide a diffusion refinement step. Avoids the slow multi-step DDPM inference used in pure diffusion methods.

**Key results:** Outperforms score-based DiffTSE on ESTOI and SI-SDR in both clean and noisy multi-speaker scenarios. In noisy multi-speaker: outperforms DiffSep+SV on all metrics. First hybrid discriminative-diffusion TSE approach.

---

### 4.2 FlowTSE — "Target Speaker Extraction with Flow Matching"
**arXiv:** 2505.14465 | **Date:** May 2025 | **Authors:** aiola lab

**Approach:** Conditional flow matching (OFM) for TSE. A 22-layer Diffusion Transformer (DiT) with 16 heads and 1024-dim embeddings processes mel-spectrograms. The mixture attends to enrollment via asymmetric cross-attention (mixed → enrollment, not vice versa). An ODE solver integrates the learned velocity field at inference. Additionally proposes a **phase-conditioned vocoder** (modified Vocos with cross-attention on mixture STFT) for better phase reconstruction.

**Training data:** LibriSpeech train-clean 100/360 + WHAM! noise, 24 kHz, 100 epochs.

**Model size:** 22 layers, 1024-dim (~200M parameters estimated).

**Metrics (Libri2Mix):**
- PESQ: up to 2.58 (highest among compared methods)
- ESTOI: 0.84
- Speaker Similarity: 0.83–0.90
- DNSMOS: 3.79–3.82
- SI-SDR: 10.81 dB (phase-conditioned vocoder variant)

**Comparison:** Outperforms discriminative baselines (DPCCN, NCSN++) and generative baselines (DiffSep, SKIM) on most perceptual metrics. Simpler pipeline than multi-component cascades.

**Code:** Audio samples at https://aiola-lab.github.io/flow-tse — full code not released.

---

### 4.3 AD-FlowTSE — "Adaptive Deterministic Flow Matching for Target Speaker Extraction"
**arXiv:** 2510.16995 | **Date:** October 2025

**Approach:** Adaptive flow matching where the transport is defined between *interference/background* and *target speech* (not noise-to-clean), controlled by the mixing ratio τ. An auxiliary ECAPA-TDNN network estimates τ at inference from mixture and enrollment, enabling adaptive step sizing per input. Uses a UNet-style Diffusion Transformer (UDiT), 16 layers, 768-dim.

**Training data:** Libri2Mix (train-360 + train-100), 16 kHz.

**Model size:** ~430M parameters (UDiT + mixing ratio predictor combined).

**Metrics (Libri2Mix):**
- SI-SDR: **17.49 dB** (noisy) / **17.73 dB** (clean, oracle mixing ratio) — among the highest reported for any generative TSE
- Speaker Similarity: 0.87–0.95
- PESQ: 2.15–2.16 (noisy) / 2.89–2.92 (clean)
- DNSMOS: 3.48–3.59

**Critical failure mode:** Random MR estimation causes severe degradation — "accurate τ estimation is critical." Fixed τ=0 (over-correct) causes artifacts; τ=1 (no correction) leaves mixture unchanged.

**Code:** https://minjekim.com/research-projects/AD-FlowTSE#icassp2026

---

### 4.4 MeanFlow-TSE — "One-Step Generative Target Speaker Extraction with Mean Flow"
**arXiv:** 2512.18572 | **