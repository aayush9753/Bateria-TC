# Training Plan: TTSE-Net End-to-End Recipe

**Date:** 2026-04-05  
**Status:** v1.0 — to be refined post-ablation  

---

## 1. DATA PIPELINE

### 1.1 Training Datasets

| Dataset | Language | Hours | Type | License | URL | Usage Stage |
|---|---|---|---|---|---|---|
| LibriSpeech train-960 | English | 960h | Clean read speech | CC-BY 4.0 | openslr.org/12 | Stage 1–4 |
| VoxCeleb2 dev | English (multi-accent) | 2,360h | Conversational | CC-BY 4.0 | robots.ox.ac.uk/~vgg/data/voxceleb | Stage 2–4 |
| LibriLight | English | 60,000h | Weakly labeled | CC-BY 4.0 | github.com/facebookresearch/libri-light | Stage 3–4 (subset) |
| VCTK | English multi-accent | 44h | Read speech | CC-BY 4.0 | datashare.ed.ac.uk | Stage 1–4 |
| DNS Challenge (ICASSP 2023) | Multilingual | 500h+ | Noisy/clean pairs | CC-BY | github.com/microsoft/DNS-Challenge | Stage 2–3 |
| Kathbath | 12 Indian langs | ~100h | Various | CC-BY 4.0 | indicvoices.ai4bharat.org | Stage 3–4 |
| Shrutilipi | 12 Indian langs | ~6400h | Broadcast | CC-BY 4.0 | indicnlp.ai4bharat.org | Stage 3–4 |
| MLS Hindi | Hindi | 50h | Read speech | CC-BY 4.0 | openslr.org/94 | Stage 3–4 |
| MUCS 2021 | Hindi-English CS | ~100h | Conversational | Research | mucs2021.ai4bharat.org | Stage 4 |
| MUSAN | Music, noise, speech | 900h | Additive noise | CC-BY 4.0 | openslr.org/17 | Noise pool |
| MusDB18-HQ | Music stems | 10h | 4-stem isolated | CC-BY-NC-SA | sigsep.github.io/musdb | Music pool |
| FSD50k | Ambient sounds | 100h | Sound events | CC-BY 4.0 | zenodo.org | Noise pool |
| BUT ReverDB | Room impulse responses | — | RIRs | CC-BY 4.0 | fit.vutbr.cz/research/groups/speech | RIR pool |
| RIR_Noises | Simulated RIRs | — | RIRs | MIT | github.com/microsoft/DNS-Challenge | RIR pool |

**Total clean speech for training:** ~2,000h English + ~2,000h Hindi/Indic (subset of Shrutilipi)

### 1.2 Mixture Simulation Pipeline

```python
# mixture_generator.py
class MixtureGenerator:
    """
    Simulates training mixtures dynamically (on-the-fly augmentation).
    Each batch generates unique mixtures — never the same mixture twice.
    """
    
    def sample_mixture(self, stage: int) -> dict:
        # 1. Sample target speaker and utterance
        target_utt = self.sample_utterance(stage=stage)
        
        # 2. Sample TTS enrollment for target
        if random.random() < self.tts_enrollment_prob[stage]:
            enrollment = self.synthesize_tts_enrollment(target_utt)
        else:
            enrollment = self.sample_clean_enrollment(target_utt.speaker_id)
        
        # 3. Sample interfering sources (stage-dependent)
        interferers = self.sample_interferers(stage=stage)
        
        # 4. Optionally add noise and music (stage-dependent)
        noise = self.sample_noise(stage=stage)    # None in Stage 1
        music = self.sample_music(stage=stage)    # None in Stages 1-2
        
        # 5. Optionally convolve with RIR (stage-dependent)
        rir = self.sample_rir(stage=stage)         # None in Stage 1
        
        # 6. Mix at random SNR (stage-dependent)
        snr = self.sample_snr(stage=stage)
        mixture = self.mix_sources(
            target=target_utt, interferers=interferers,
            noise=noise, music=music, rir=rir, snr=snr
        )
        
        return {
            'mixture': mixture,
            'target': target_utt.waveform,
            'enrollment': enrollment,
            'transcript': target_utt.transcript
        }
```

### 1.3 TTS Enrollment Synthesis During Training

**Strategy:** Not all training samples use TTS enrollment. Mix real and TTS enrollment:

| Stage | TTS Enrollment Probability | TTS System |
|---|---|---|
| Stage 1 | 0% (real only) | — |
| Stage 2 | 30% | XTTS-v2 |
| Stage 3 | 60% | XTTS-v2 (70%), YourTTS (30%) |
| Stage 4 | 70% | XTTS-v2 (50%), YourTTS (30%), OpenVoice (20%) |

**Pre-synthesis for efficiency:** Pre-synthesize TTS enrollments for all training speakers offline (cached to disk). At runtime, load pre-synthesized enrollment WAV instead of running TTS inference.

**Pre-synthesis compute:** 
- LibriSpeech: ~2,000 unique speakers × ~10 utterances each = 20,000 TTS calls ≈ 11h on 2×A100
- Hindi/Indic: ~2,000 speakers × 5 utterances = 10,000 calls ≈ 5.5h on 2×A100

### 1.4 Augmentation Strategy

```python
augmentation_pipeline = [
    # Speed perturbation (0.9x, 1.0x, 1.1x) — 50% probability
    SpeedPerturb(rates=[0.9, 1.0, 1.1], prob=0.5),
    
    # Codec simulation (MP3, AAC, Opus) — 30% probability
    # Models the degradation of streaming audio
    CodecSim(codecs=['mp3_64k', 'opus_24k', 'aac_64k'], prob=0.3),
    
    # Telephone/telephony bandwidth (300–3400 Hz) — 20% probability (conversational data)
    BandwidthLimit(low=300, high=3400, prob=0.2),
    
    # Microphone/headset simulation (convolution with headset IR) — 25% probability
    MicSim(mic_types=['headset', 'smartphone', 'laptop'], prob=0.25),
    
    # Volume perturbation (0.5–2.0× scaling) — always
    VolumePerturb(range=(0.5, 2.0), prob=1.0),
    
    # DC offset — 10% probability
    DCOffset(max_offset=0.01, prob=0.1),
]
```

### 1.5 Train / Val / Test Split

| Split | Purpose | Data |
|---|---|---|
| Train | Model training | Everything except val/test speakers |
| Val (English) | Hyperparameter tuning | 50 LibriSpeech speakers (held out from train) |
| Val (Hindi) | Multilingual validation | 20 Kathbath speakers (held out) |
| Test | Final evaluation only | Full test set (test_set_design.md) |

**Speaker non-overlap:** Strictly maintained. No speaker in val/test appears in training data.

---

## 2. CURRICULUM LEARNING

### Stage 1: Foundation — Clean English 2-Speaker

**Goal:** Model learns basic speaker-conditioned separation mechanics  
**Duration:** 200,000 steps  
**Batch size:** 16  
**Learning rate:** 1e-3 (Adam, β1=0.9, β2=0.999)  
**Dataset mix:**
  - Target: LibriSpeech train-960 + VCTK (80/20)
  - Interferer: LibriSpeech train-360 (different speakers)
  - n_speakers: 2
  - SNR: [0, 10] dB
  - Noise: None
  - Music: None
  - Reverb: None
  - TTS enrollment: 0% (real enrollment only)

**Validation metric to advance:** Val SI-SDRi > 14 dB on EN-C1 condition

### Stage 2: Noise & Enrollment Robustness

**Goal:** Learn to separate in noisy conditions; begin TTS enrollment training  
**Duration:** 150,000 steps  
**Batch size:** 16  
**Learning rate:** 5e-4  
**Dataset mix:**
  - Target: LibriSpeech + VCTK + VoxCeleb2 (conversational)
  - Interferer: LibriSpeech + VoxCeleb2
  - n_speakers: 2–3 (80% 2-spk, 20% 3-spk)
  - SNR: [-5, 10] dB
  - Noise: MUSAN noise (50% probability)
  - Music: None
  - Reverb: 30% probability (BUT ReverDB)
  - TTS enrollment: 30%

**Validation metric to advance:** Val SI-SDRi > 12 dB on EN-N1 condition with TTS enrollment

### Stage 3: Multilingual + Music

**Goal:** Extend to Hindi and add music background handling  
**Duration:** 150,000 steps  
**Batch size:** 24 (increase GPU utilization)  
**Learning rate:** 3e-4  
**Dataset mix (proportional):**
  - English speech: 50% (LibriSpeech + VoxCeleb2)
  - Hindi speech: 30% (Shrutilipi + MLS Hindi + Kathbath)
  - Code-switched: 10% (MUCS 2021)
  - Other (DNS multilingual): 10%
  - n_speakers: 2–3
  - SNR: [-5, 10] dB
  - Noise: MUSAN noise + MUSAN music (40% each, independently)
  - Music: MusDB18-HQ non-vocal (30% probability)
  - Reverb: 40% probability
  - TTS enrollment: 60%

**Validation metric to advance:**
  - HI-C1 SI-SDRi > 11 dB
  - EN-M1 SI-SDRi > 10 dB

### Stage 4: Hard Conditions + Full Distribution

**Goal:** Final generalization across all conditions  
**Duration:** 100,000 steps  
**Batch size:** 32  
**Learning rate:** 1e-4 (cosine decay to 1e-5)  
**Dataset mix:**
  - All languages: same proportions as Stage 3
  - n_speakers: 2–4 (60% 2, 30% 3, 10% 4)
  - SNR: [-10, 10] dB (harder)
  - Noise: All noise sources
  - Music: 40% probability (more frequent)
  - Reverb: 50% probability
  - TTS enrollment: 70%
  - Codec augmentation: enabled

**Validation metric:** All conditions improve or maintain vs. Stage 3 checkpoint

---

## 3. LOSS FUNCTIONS

### 3.1 Primary: Scale-Invariant SDR (SI-SDR)

```python
def si_sdr_loss(estimated, target):
    """
    SI-SDR loss. Scale-invariant version of SDR.
    Negative because we maximize SI-SDR.
    """
    alpha = (target * estimated).sum() / (target * target).sum()
    signal = alpha * target
    noise = estimated - signal
    si_sdr = 10 * torch.log10(signal.pow(2).sum() / noise.pow(2).sum())
    return -si_sdr.mean()
```

**Weight:** 1.0

### 3.2 Speaker Similarity Loss

```python
def speaker_similarity_loss(extracted_emb, target_emb):
    """
    Cosine similarity loss between speaker embeddings.
    Drives extracted speech to sound like the target speaker.
    """
    cos_sim = F.cosine_similarity(extracted_emb, target_emb, dim=-1)
    return (1 - cos_sim).mean()

# extracted_emb: ECAPA-TDNN(extracted_speech)
# target_emb: ECAPA-TDNN(clean_target_speech) — from ground-truth
```

**Weight Schedule:**
- Stage 1: 0.0 (not applied — too early, can destabilize)
- Stage 2: 0.1
- Stage 3: 0.3
- Stage 4: 0.5

### 3.3 STFT Magnitude Loss (Spectral Consistency)

```python
def stft_loss(estimated, target):
    """
    Multi-resolution STFT magnitude loss.
    Helps with frequency-domain quality, especially for music-contaminated audio.
    """
    losses = []
    for fft_size, hop_size in [(512, 128), (1024, 256), (2048, 512)]:
        est_stft = torch.stft(estimated, fft_size, hop_size, return_complex=True)
        tgt_stft = torch.stft(target, fft_size, hop_size, return_complex=True)
        mag_loss = F.l1_loss(est_stft.abs(), tgt_stft.abs())
        losses.append(mag_loss)
    return sum(losses) / len(losses)
```

**Weight:** 0.2 (all stages)

### 3.4 ASR CTC Auxiliary Loss (Stage 3+)

```python
def asr_ctc_loss(extracted_speech, transcripts):
    """
    Auxiliary CTC loss using frozen Whisper encoder.
    Drives intelligibility of extracted speech.
    """
    with torch.no_grad():  # Only fine-tune via gradient clipping, not backprop through whisper
        whisper_enc = whisper_model.encode(extracted_speech)
    ctc_logits = ctc_head(whisper_enc)
    return F.ctc_loss(ctc_logits, transcripts, ...)
```

**Weight Schedule:**
- Stage 1–2: 0.0
- Stage 3: 0.05
- Stage 4: 0.1

**Note:** CTC loss helps prevent over-fitting to signal quality at the expense of intelligibility, especially for code-switched audio.

### 3.5 Total Loss

```python
total_loss = (
    1.0 * si_sdr_loss(estimated, target)
    + 0.2 * stft_loss(estimated, target)
    + lambda_spk * speaker_similarity_loss(ECAPA(estimated), ECAPA(target))
    + lambda_asr * asr_ctc_loss(estimated, transcript)
)
# lambda_spk, lambda_asr follow schedule above
```

---

## 4. OPTIMIZER & LEARNING RATE

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=initial_lr,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5
)

# Learning rate schedule (Stage 4):
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100_000,
    eta_min=1e-5
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### Warmup (Stage 1 only)
- Linear warmup from 1e-5 → 1e-3 over 10,000 steps

---

## 5. INFRASTRUCTURE

### 5.1 Hardware Requirements

| Stage | GPUs | Est. Wall Time | GPU-Hours |
|---|---|---|---|
| Speaker encoder pre-training/fine-tuning | 2× A100 40GB | 2 days | 96h |
| TTS pre-synthesis (enrollment) | 2× A100 40GB | 2 days | 96h |
| Stage 1 (200k steps) | 4× A100 40GB | 2 days | 192h |
| Stage 2 (150k steps) | 4× A100 40GB | 1.5 days | 144h |
| Stage 3 (150k steps) | 4× A100 40GB | 2 days | 192h |
| Stage 4 (100k steps) | 4× A100 40GB | 1.5 days | 144h |
| **Total** | — | **~11 days** | **~864 GPU-hours** |

*Equivalent cost on cloud (A100 @ $2/hr): ~$1,728 USD per training run*

### 5.2 Mixed Precision & Memory

```python
# PyTorch AMP (automatic mixed precision)
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(mixture, enrollment)
    loss = compute_loss(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Gradient checkpointing:** Enabled for conformer layers (saves ~30% GPU memory; ~15% slower).

### 5.3 Distributed Training

```bash
# 4-GPU DDP training
torchrun --nproc_per_node=4 train.py \
    --config configs/stage1.yaml \
    --data_root /data/speech \
    --checkpoint_dir /checkpoints/ttse_net
```

**Effective batch size:** 4 GPUs × 16 samples = 64 samples per step.

### 5.4 Checkpoint Evaluation

```
Every 10,000 steps: save checkpoint + evaluate on English validation set
Every 25,000 steps: full evaluation on all validation conditions
Best checkpoint (by EN-C1 SI-SDRi): saved as "best_en.pt"
Best checkpoint (by HI-C1 SI-SDRi): saved as "best_hi.pt"
Best checkpoint (by average all): saved as "best_overall.pt"
```

### 5.5 Experiment Tracking

```yaml
# W&B (Weights & Biases) configuration
project: ttse_net
entity: speech_ai_team
tags: [tts_guided, multilingual, music_aware]

logged_metrics:
  - train/loss_si_sdr
  - train/loss_stft  
  - train/loss_speaker_sim
  - val_en/si_sdri_clean
  - val_en/si_sdri_noisy
  - val_en/si_sdri_music
  - val_hi/si_sdri_clean
  - val_cs/si_sdri_clean
  - grad_norm
  - lr
```

---

## 6. TRAINING CODE STRUCTURE

```
ttse_net/
├── configs/
│   ├── stage1.yaml
│   ├── stage2.yaml
│   ├── stage3.yaml
│   └── stage4.yaml
├── data/
│   ├── mixture_generator.py   # On-the-fly mixture simulation
│   ├── dataset.py             # PyTorch Dataset classes
│   ├── tts_enrollment.py      # TTS synthesis pipeline
│   └── augmentation.py        # Audio augmentation ops
├── models/
│   ├── ttse_net.py            # Main model
│   ├── speaker_encoder.py     # ECAPA-TDNN wrapper
│   ├── tf_template.py         # T-F template extractor
│   ├── band_split.py          # Band-split module
│   ├── conformer_lstm.py      # Separation backbone
│   └── decoder.py             # ISTFT + OLA decoder
├── losses/
│   ├── si_sdr.py
│   ├── stft_loss.py
│   ├── speaker_sim.py
│   └── asr_ctc.py
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
└── infer.py                   # Inference script (offline + streaming)
```

---

## 7. FAILURE MODES & MITIGATIONS

| Risk | Likelihood | Mitigation |
|---|---|---|
| TTS enrollment degrading quality vs. real | Medium | Train with mixed real/TTS; robustness fine-tuning |
| Hindi speaker encoder weak | High | Fine-tune ECAPA on IndicSUPERB speaker ID |
| Music backgrounds causing hallucination | Medium | BSRNN-style bands; STFT loss |
| Overfitting to synthesis artifacts | Medium | Multiple TTS systems; codec augmentation |
| Code-switching encoder confusion | Medium | Explicitly include CS data in Stage 3+ |
| Long-form audio degradation | Medium | Random chunking; long context testing |
| Mode collapse (all outputs same speaker) | Low | Batch diversity enforcement; speaker balance in batches |

---

## 8. MODEL VARIANTS PLAN

| Variant | Architecture Changes | Target Use Case |
|---|---|---|
| TTSE-Net-L (Large) | 6→12 conformer layers | Maximum quality, offline |
| TTSE-Net-B (Base) | As designed (~34M) | Standard production |
| TTSE-Net-S (Small) | Unidirectional LSTM, no cross-band conformer | Streaming, on-device |
| TTSE-Net-M (Music) | Extra music-specific loss, MusDB fine-tune | Music-heavy environments |
| TTSE-Net-EN (English-only) | English training only, smaller | English-only deployments |
