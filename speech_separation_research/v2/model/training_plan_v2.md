# Training Plan v2: TTSE-Net (Unconstrained)
**Date:** 2026-04-06  
**Assumptions:** No compute/size/data constraints. Can purchase data. Train from scratch.

---

## 1. TOTAL DATA STRATEGY

### 1.1 Clean Speech (Target Speaker Sources)

| Dataset | Language | Hours | License | Source | Notes |
|---|---|---|---|---|---|
| LibriLight | English | 60,000 | CC-BY 4.0 | Facebook/Librivox | Pre-training scale |
| EMILIA (English subset) | English | ~50,000 | CC-BY 4.0 | arXiv 2407.05361 | In-the-wild, filtered by DNSMOS>3.4 |
| GigaSpeech XL | English | 10,000 | Apache 2.0 | openslr.org/86 | YouTube/podcast |
| MLS English | English | 44,500 | CC-BY 4.0 | openslr.org/94 | Read speech |
| VoxPopuli | En+14 langs | 400,000 | CC-0 | Facebook | Multilingual, parliament |
| Common Voice 17 | 100+ langs | 30,000+ | CC-0 | Mozilla | Community recordings |
| Shrutilipi | Hindi+11 langs | 6,400 | CC-BY 4.0 | AI4Bharat | Broadcast, Indian languages |
| Kathbath | 12 Indian langs | ~1,000 | CC-BY 4.0 | AI4Bharat | Natural reading |
| MUCS 2021 | Hindi-English CS | ~100 | Research | AI4Bharat | Code-switched |
| MLS Hindi | Hindi | 50 | CC-BY 4.0 | openslr.org/94 | Read speech |
| EMILIA (Hindi/multilingual) | Hindi+others | ~5,000 | CC-BY 4.0 | AI4Bharat | In-the-wild |
| Switchboard+Fisher | English conversational | 2,000 | **Purchase LDC** | LDC2004S13 | Natural conversation |
| CallHome | English/Spanish | 120 | **Purchase LDC** | LDC2001S97 | Phone conversation |
| LibriheavyMix (source audio) | English | 20,000 | CC-BY 4.0 | arXiv 2409.00819 | Multi-turn, reverberant |
| WenetSpeech | Mandarin | 10,000 | CC-BY 4.0 | github.com/wenet-e2e | In-the-wild Chinese |
| VoxCeleb2 dev | English multi-accent | 2,360 | CC-BY 4.0 | VoxCeleb | Celebrity, diverse |
| VCTK | English multi-accent | 44 | CC-BY 4.0 | U. Edinburgh | 110 speakers, controlled |

**Total clean speech: ~250,000+ hours** (dominated by EMILIA + VoxPopuli + LibriLight)

### 1.2 Music Sources

| Dataset | Hours | License | Notes |
|---|---|---|---|
| MusDB18-HQ | 10 | CC-BY-NC-SA 4.0 | 4-stem isolated (vocals, drums, bass, other) |
| MUSAN music | 42 | CC-BY 4.0 | Various genres |
| FMA-Large | 8,000 | CC-BY | Free Music Archive, diverse genres |
| Freesound CC music | ~500 | CC-BY/0 | Via Freesound API, batch download |
| YouTube AudioSet music clips | 5,000+ | Fair use/research | ~10s clips, 600 music categories |

**Total music: ~14,000 hours**

### 1.3 Noise / Room Impulse Responses

| Dataset | Type | Size | License |
|---|---|---|---|
| MUSAN | Noise + music + speech | 900h | CC-BY 4.0 |
| FSD50k | Sound events | 51h | CC-BY 4.0 |
| DEMAND | Real-world noise | 3h | CC-BY-SA |
| DNS Challenge (ICASSP 2023) | Mixed noise | 500h+ | CC-BY |
| BUT ReverbDB | Room impulse responses | 8,000 RIRs | CC-BY 4.0 |
| OpenSLR RIR (SLR26, SLR28) | RIRs | 60,000 RIRs | Apache 2.0 |
| Aachen IR Database | RIRs | 1,000 RIRs | Free |
| URGENT Challenge noise pool | Real-world noise | 1,000h | Mixed CC |

### 1.4 TTS Enrollment Pre-Synthesis (Training)

**Strategy:** Pre-synthesize TTS enrollments for all training speakers and cache to disk. Never run TTS at training time.

**Execution plan:**
```
For each training speaker:
  1. Select a 5-second reference clip (distinct from training utterances)
  2. Extract voice clone embedding via XTTS-v2
  3. Synthesize 3 enrollments per speaker (different transcripts)
  4. Repeat with F5-TTS and CosyVoice (3 TTS systems)
  5. Cache all 9 synthetic enrollments to disk

Total unique speakers: ~500,000 (across all datasets)
Total TTS calls: ~500,000 × 9 = 4.5M TTS synthesizations
Estimated time: 4.5M × 2s / (8 A100 GPUs) ≈ 130 hours
```

**Hindi-specific:** Use CosyVoice (supports Hindi, Chinese, etc.) and XTTS-v2 (supports Hindi via multilingual model).

---

## 2. MIXTURE SIMULATION

### 2.1 On-the-Fly Dynamic Mixing
All mixtures generated dynamically during training — no fixed train set.

```python
class DynamicMixtureSampler:
    """
    On-the-fly mixture generation. Each training step sees a unique mixture.
    Designed for training TTSE-Net v2.
    """
    def __init__(self, stage: int, config: dict):
        self.stage = stage
        self.n_speakers = config['n_speakers']       # 2-4
        self.snr_range = config['snr_range']         # dB
        self.music_prob = config['music_prob']       # 0.0 - 0.5
        self.noise_prob = config['noise_prob']       # 0.0 - 0.8
        self.reverb_prob = config['reverb_prob']     # 0.0 - 0.7
        self.tts_enrollment_prob = config['tts_prob']# 0.0 - 0.8
        self.codec_aug_prob = config['codec_prob']   # 0.0 - 0.4

    def sample(self, speech_pool, music_pool, noise_pool, rir_pool, tts_cache):
        # 1. Sample target utterance (3-15 seconds)
        target = speech_pool.sample()
        
        # 2. Sample enrollment
        if random() < self.tts_enrollment_prob:
            # TTS enrollment — randomly choose TTS system and transcript variant
            enrollment = tts_cache.sample(target.speaker_id,
                                          tts_system=choice(['xtts', 'f5', 'cosyvoice']))
        else:
            # Real enrollment — different utterance from same speaker
            enrollment = speech_pool.sample_same_speaker(target.speaker_id,
                                                          exclude=target.id)
        
        # 3. Sample interferers
        n_inter = randint(1, self.n_speakers - 1)
        interferers = [speech_pool.sample_different_speaker(target.speaker_id)
                       for _ in range(n_inter)]
        
        # 4. Optional music (always use isolated stems for clean mixing)
        music = music_pool.sample() if random() < self.music_prob else None
        
        # 5. Optional noise
        noise = noise_pool.sample() if random() < self.noise_prob else None
        
        # 6. Optional RIR (applied to each source independently)
        rir = rir_pool.sample() if random() < self.reverb_prob else None
        
        # 7. Mix at random SNR
        mixture, ref_clean = mix_sources(
            target=target,
            interferers=interferers,
            music=music,
            noise=noise,
            rir=rir,
            snr=uniform(*self.snr_range),
            sample_rate=24000,
        )
        
        # 8. Codec degradation augmentation
        if random() < self.codec_aug_prob:
            mixture = apply_codec(mixture, codec=choice(['mp3_64k', 'opus_24k', 'aac_64k']))
        
        return {
            'mixture': mixture,          # (T,) float32 at 24kHz
            'target_clean': ref_clean,   # (T,) float32 at 24kHz
            'enrollment': enrollment,    # (T_ref,) float32 at 24kHz
            'language': target.language,
            'has_music': music is not None,
        }
```

### 2.2 SNR Ranges by Stage

| Stage | SNR range | n_speakers | Music prob | Noise prob | Reverb prob | TTS enrollment |
|---|---|---|---|---|---|---|
| 1 | [0, 10] dB | 2 | 0% | 0% | 0% | 0% |
| 2 | [-5, 10] dB | 2-3 | 0% | 50% | 30% | 30% |
| 3 | [-5, 10] dB | 2-4 | 20% | 60% | 50% | 60% |
| 4 | [-10, 10] dB | 2-4 | 40% | 70% | 60% | 70% |
| 5 | [-15, 10] dB | 2-5 | 50% | 80% | 70% | 80% |

---

## 3. CURRICULUM (5 STAGES)

### Stage 1 — Foundation: English Clean 2-Speaker
**Goal:** Learn basic CMHA conditioning and separation mechanics  
**Steps:** 200,000  
**GPUs:** 8× H100 80GB  
**Batch size:** 32 (effective: 256 with gradient accumulation)  
**LR:** 1e-4 (warmup 10k steps → 5e-4 → cosine decay)  

**Data:**
- English only: LibriSpeech + VCTK + MLS
- 2 speakers, clean, no noise/music/reverb
- 100% real enrollment (build baseline speaker conditioning)
- WavLM frozen; only CMHA + backbone trained

**Loss:** SI-SDR only  
**Advance when:** Val EN-C1 SI-SDRi > 17 dB

---

### Stage 2 — Robustness: Noise + Reverb + TTS Enrollment
**Goal:** Handle degraded conditions and begin TTS enrollment  
**Steps:** 200,000  
**GPUs:** 8× H100  
**Batch size:** 64  
**LR:** 3e-4  

**Data:**
- English 80%, other 20%
- Noise (MUSAN), reverb (BUT DB)
- 30% TTS enrollment, 70% real
- Begin unfreezing WavLM top 4 layers at step 100k

**Loss:** SI-SDR + multi-resolution STFT (0.2×)  
**Advance when:** Val EN-N1 SI-SDRi > 14 dB with TTS enrollment

---

### Stage 3 — Multilingual: Hindi + Code-Switched
**Goal:** Generalize to Indian languages  
**Steps:** 250,000  
**GPUs:** 16× H100 (scale up)  
**Batch size:** 128  
**LR:** 2e-4  

**Data per batch:**
- English: 50%
- Hindi (Shrutilipi + Kathbath + MLS): 30%
- Code-switched Hindi-English (MUCS): 10%
- Other multilingual (VoxPopuli, Common Voice): 10%
- 60% TTS enrollment
- Unfreeze WavLM fully at step 50k

**Loss:** SI-SDR + STFT + Speaker similarity (WavLM cosine, 0.3×)  
**Advance when:** Val HI-C1 SI-SDRi > 14 dB, EN-N1 SI-SDRi maintained > 14 dB

---

### Stage 4 — Music: Joint Speech+Music Separation
**Goal:** Handle music backgrounds without degrading speech extraction  
**Steps:** 200,000  
**GPUs:** 16× H100  
**Batch size:** 128  
**LR:** 1e-4  

**Data:** All languages, now 40% of batches include music (MusDB18 + FMA + MUSAN music)  
**Special loss:** Add music residual loss (0.1×) — encourages clean music in residual  
**70% TTS enrollment**

**Advance when:** Val EN-M1 SI-SDRi > 12 dB, speaker similarity maintained

---

### Stage 5 — Hard Conditions: All Combined + Scale
**Goal:** Final generalization push  
**Steps:** 150,000  
**GPUs:** 16× H100  
**Batch size:** 256  
**LR:** 5e-5 → cosine decay to 1e-5  

**Data:** All conditions, hardest SNR range (-15 to +10 dB), 2-5 speakers, all augmentations  
**80% TTS enrollment**  
**Add DPO fine-tuning (Stage 5b, 50k steps):** Collect perceptual preference rankings, apply DPO  

---

## 4. LOSS FUNCTIONS

### 4.1 Primary: SI-SDR
```python
def si_sdr_loss(estimate, target, eps=1e-8):
    # Scale-invariant SDR. estimate, target: (B, T)
    target_norm = (target * estimate).sum(-1, keepdim=True) / \
                  (target.pow(2).sum(-1, keepdim=True) + eps)
    target_proj = target_norm * target
    noise = estimate - target_proj
    si_sdr = 10 * torch.log10(
        target_proj.pow(2).sum(-1) / (noise.pow(2).sum(-1) + eps) + eps
    )
    return -si_sdr.mean()
```

### 4.2 Multi-Resolution STFT Loss
```python
def mrstft_loss(estimate, target, fft_sizes=[512, 1024, 2048, 4096]):
    """Multi-resolution STFT magnitude + log magnitude loss."""
    total = 0
    for fft_size in fft_sizes:
        hop = fft_size // 4
        est_stft = torch.stft(estimate, fft_size, hop, return_complex=True)
        tgt_stft = torch.stft(target, fft_size, hop, return_complex=True)
        # Magnitude L1
        total += F.l1_loss(est_stft.abs(), tgt_stft.abs())
        # Log-magnitude L1 (emphasizes quiet regions — important for music)
        total += F.l1_loss(
            torch.log(est_stft.abs() + 1e-8),
            torch.log(tgt_stft.abs() + 1e-8)
        )
    return total / len(fft_sizes)
```

### 4.3 Speaker Similarity Loss (Stages 3+)
```python
def speaker_sim_loss(estimate, target):
    """
    Cosine similarity between WavLM speaker representations.
    Uses WavLM-base-plus (faster than large) for in-training loss computation.
    """
    with torch.no_grad():
        est_feat = wavlm_base(estimate).last_hidden_state.mean(1)  # (B, 768)
        tgt_feat = wavlm_base(target).last_hidden_state.mean(1)    # (B, 768)
    est_feat = F.normalize(est_feat, dim=-1)
    tgt_feat = F.normalize(tgt_feat, dim=-1)
    return (1 - (est_feat * tgt_feat).sum(-1)).mean()
```

### 4.4 Music Residual Loss (Stage 4+)
```python
def music_residual_loss(mixture, estimate, music_ref):
    """
    Encourage the residual (mixture - estimated_speech) to sound like music.
    Only applied when batch has music background.
    """
    residual = mixture - estimate  # What we left in the mixture
    # SDR between residual and original music
    return si_sdr_loss(residual, music_ref)
```

### 4.5 ASR CTC Auxiliary Loss (Stage 3+)
```python
def asr_ctc_loss(estimate, transcripts, tokenizer):
    """
    CTC loss using frozen Whisper encoder to drive intelligibility.
    Gradient does NOT flow through Whisper — only through the extraction model.
    """
    with torch.no_grad():
        enc = whisper_encoder(estimate)  # (B, T', 1280) for large-v3
    logits = ctc_projection(enc)         # (B, T', vocab_size)
    targets, target_lengths = tokenizer.encode_batch(transcripts)
    return F.ctc_loss(logits.transpose(0,1), targets, 
                      estimate.shape[-1] // 320,  # input lengths
                      target_lengths)
```

### 4.6 Total Loss Schedule

```
Stage 1: L = SI-SDR
Stage 2: L = SI-SDR + 0.2 × MRSTFT
Stage 3: L = SI-SDR + 0.2 × MRSTFT + 0.3 × Speaker-Sim + 0.05 × CTC
Stage 4: L = SI-SDR + 0.2 × MRSTFT + 0.4 × Speaker-Sim + 0.05 × CTC + 0.1 × Music-Residual
Stage 5: L = SI-SDR + 0.2 × MRSTFT + 0.5 × Speaker-Sim + 0.1 × CTC + 0.1 × Music-Residual
```

---

## 5. INFRASTRUCTURE

### 5.1 Hardware Plan

| Stage | GPUs | Type | Wall Time | GPU-Hours |
|---|---|---|---|---|
| TTS pre-synthesis | 8× A100 40GB | — | 4 days | 768h |
| WavLM Speaker Encoder FT | 4× H100 80GB | — | 2 days | 192h |
| Stage 1 | 8× H100 80GB | FSDP | 3 days | 576h |
| Stage 2 | 8× H100 80GB | FSDP | 3 days | 576h |
| Stage 3 | 16× H100 80GB | FSDP | 5 days | 1,920h |
| Stage 4 | 16× H100 80GB | FSDP | 4 days | 1,536h |
| Stage 5 | 16× H100 80GB | FSDP | 3 days | 1,152h |
| DPO (Stage 5b) | 8× H100 | — | 3 days | 576h |
| Track 2 (DiT, 2.5B) | 32× H100 | FSDP+TP | 14 days | 10,752h |
| **Track 1 Total** | — | — | **~27 days** | **~7,300h** |
| **Track 2 Total** | — | — | **~14 days** | **~10,750h** |

*Assuming H100 80GB at $4/hr cloud rate: Track 1 ≈ $29,200; Track 2 ≈ $43,000*

### 5.2 Distributed Training Config

```python
# Fully Sharded Data Parallel (FSDP) for memory efficiency on large models
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# For Track 1 (~750M params on 8× H100):
# Each GPU holds 750M/8 = ~94M params (after FSDP sharding)
# Plus activations: ~40GB per GPU → fits in 80GB H100

# Mixed precision: bfloat16 throughout (H100 native)
torch.set_default_dtype(torch.bfloat16)

# Gradient checkpointing for TF-GridNet blocks (trades compute for memory)
# Saves ~30% memory at ~20% compute cost

# Data loading: 16 workers per GPU, pinned memory, prefetch_factor=4
```

### 5.3 Checkpoint Strategy

```
Every 10,000 steps:  Save checkpoint + quick eval on English val set
Every 25,000 steps:  Full evaluation on all val conditions
Every stage end:     Full benchmark eval + ablation report

Checkpoints saved:
  best_en_sdr.pt         (highest EN-C1 SI-SDRi)
  best_hi_sdr.pt         (highest HI-C1 SI-SDRi)
  best_speaker_sim.pt    (highest WavLM similarity)
  best_overall.pt        (weighted average)
  latest.pt              (always latest)
```

---

## 6. EVALUATION DURING TRAINING

**Val sets** (held out from training, fixed):

| Split | Size | Purpose |
|---|---|---|
| EN-val-clean | 200 samples | English 2-spk clean — tracks basic quality |
| EN-val-noisy | 200 samples | English noisy — tracks noise robustness |
| EN-val-music | 200 samples | English + music — tracks music awareness |
| HI-val-clean | 100 samples | Hindi clean — tracks multilingual progress |
| CS-val | 100 samples | Code-switched — tracks CS progress |
| TTS-gap | 100 samples | Same conditions: real vs. TTS enrollment gap |

**Key metric to minimize:** TTS-gap (SI-SDR difference between real and TTS enrollment on EN-val-clean)

Target: TTS-gap < 1 dB throughout training.

---

## 7. ITERATIVE REFINEMENT (INFERENCE-TIME)

For offline high-quality inference, add 1-2 refinement passes:

```python
def refined_extraction(model, mixture, tts_reference, n_iters=2):
    """
    Iterative refinement: re-use extracted speech as better enrollment.
    
    Iter 0: Use TTS reference → extract ŝ_0
    Iter 1: Use ŝ_0 as enrollment (now correct timing) → extract ŝ_1
    Iter 2: Use ŝ_1 as enrollment → extract ŝ_2 (usually final)
    """
    enrollment = tts_reference
    for i in range(n_iters + 1):
        extracted = model(mixture, enrollment)
        # Quality gate: only refine if quality is acceptable
        if compute_dnsmos(extracted) < 2.5:
            break  # TTS reference was too bad; don't iterate
        enrollment = extracted  # Use extracted as next enrollment
    return extracted
```

**Expected gain per iteration:** ~0.5–1.5 dB SI-SDR (to be validated in ablation).  
**Max 2 iterations** — diminishing returns beyond this.
