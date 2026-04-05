# Ablation Plan: Existing SOTA Models on Our Test Set

**Date:** 2026-04-05  
**Purpose:** Systematically evaluate the top 5 SOTA models to understand failure modes before designing our own system  

---

## OVERVIEW

Running these ablations **before designing our model** serves to:
1. Establish concrete baselines
2. Identify which conditions each paradigm handles well/poorly
3. Inform architecture choices (what the base model must improve upon)
4. Estimate compute and pipeline requirements

All evaluations use the test set defined in `test_set_design.md`.

---

## MODEL 1: SpEx+ (Speaker-Conditioned TSE)

### Setup
```bash
# Install
pip install torch torchaudio asteroid

# Or clone directly
git clone https://github.com/xuchenglin28/speaker_extraction
pip install -r requirements.txt

# Model checkpoint: pre-trained on WSJ0-2mix
# Download from: [see README]
```

**Input format:**
- Mixture: 16kHz mono WAV, any duration
- Enrollment: 16kHz mono WAV, 2–10 seconds
- Output: 16kHz mono WAV (extracted speech)

**Enrollment sources to test (3 variants per sample):**
1. Real enrollment: a clean clip of target speaker from the same dataset
2. TTS enrollment (XTTS-v2): synthesized with voice clone
3. TTS enrollment (YourTTS): synthesized with voice clone

**Test Conditions to Run:**
| Condition | Priority | Expected |
|---|---|---|
| EN-C1 (clean 2-spk) | High | Strong: trained on this condition |
| EN-N1 (noisy 2-spk) | High | Moderate: noise degrades speaker encoder |
| EN-M1 (music background) | High | Weak: never trained on music |
| EN-A1 (all combined) | Medium | Very weak |
| HI-C1 (Hindi clean 2-spk) | High | **Very weak**: English-only training |
| CS-C1 (code-switched) | Medium | Very weak |

**What we expect to learn:**
- Gap between real enrollment and TTS enrollment (quantify in SI-SDR and speaker similarity)
- How much noise degrades speaker conditioning (is the speaker encoder the bottleneck?)
- Catastrophic failure on Hindi → confirms multilingual gap exists
- Music failure mode → confirms need for music-aware backbone

**Compute estimate:**
- 6 conditions × ~600 samples = 3,600 inference runs
- ~0.5s per sample on A100 = ~30 min
- 3 enrollment variants = 90 min total

---

## MODEL 2: SepFormer (Blind Separation Upper Bound)

### Setup
```python
from speechbrain.pretrained import SepformerSeparation as sep
model = sep.from_hparams("speechbrain/sepformer-wsj02mix")
# OR for noisy: "speechbrain/sepformer-wham"

# Input: mixture WAV (16kHz)
# Output: list of separated speaker WAVs
# Note: output order is arbitrary (permutation ambiguity) — use speaker similarity for alignment
```

**Key challenge:** Blind separation = permutation ambiguity. For our test set with a known target speaker, we select the output with highest ECAPA cosine similarity to the target.

**Test Conditions:**
| Condition | Priority | Expected |
|---|---|---|
| EN-C1 (clean 2-spk) | High | Strong (trained condition) |
| EN-C2 (clean 3-spk) | High | Moderate to strong |
| EN-N1 (noisy 2-spk) | High | Use sepformer-wham variant |
| EN-M1 (music) | Medium | Weak: separates music as a "speaker" |
| HI-C1 (Hindi clean) | High | Moderate: language-agnostic to some extent |
| CS-C1 (code-switched) | Medium | Moderate |

**What we expect to learn:**
- This is the *maximum signal quality* achievable with blind separation (no speaker conditioning)
- Gap vs. SpEx+ shows value of speaker conditioning
- Permutation ambiguity rate: how often does it output the wrong speaker?
- Whether blind separation handles Hindi better than speaker-conditioned English-trained models

**Compute estimate:**
- 6 conditions × ~600 = 3,600 runs × 2 models (wsj02mix + wham) = ~60 min A100

---

## MODEL 3: TF-GridNet (T-F Domain Quality Ceiling)

### Setup
```bash
# Best available: ESPnet
pip install espnet espnet2

# Or: https://github.com/Wenzhe-Liu/TF-GridNet (unofficial)
# Config: standard 2-speaker model, WSJ0-2mix

# Input: 16kHz WAV
# Output: 2 separated WAVs
```

**Note:** TF-GridNet has no official speaker conditioning. We will:
1. Run blind separation; post-select with ECAPA similarity
2. Optionally: test naive speaker conditioning by concatenating d-vector to LSTM hidden state

**Test Conditions:**
| Condition | Priority | Expected |
|---|---|---|
| EN-C1 (clean 2-spk) | High | Best performance (23.4 dB trained condition) |
| EN-N1 (noisy 2-spk) | High | Strong: tested on WHAM! |
| EN-M1 (music) | High | Moderate: T-F domain excels on structured interference |
| EN-R1 (reverberant) | Medium | Good: T-F domain handles reverb well |
| HI-C1 (Hindi) | Medium | Moderate (language-agnostic architecture) |

**What we expect to learn:**
- Absolute SI-SDR ceiling on T-F domain models
- How T-F domain handles music vs. TCN-based models
- Whether blind T-F separation handles Hindi as well as English (architecture generalization)
- **Key insight:** Compare music performance of TF-GridNet vs. SpEx+ to see if backbone choice matters more than conditioning

**Compute estimate:** ~45 min A100 for all conditions.

---

## MODEL 4: BSRNN (Music-Aware Separation)

### Setup
```bash
git clone https://github.com/popcornell/BSRNN
pip install -r requirements.txt
# Use model trained on WSJ0+MusDB18 hybrid (best for music-aware)

# Input: 44.1kHz or 16kHz (check model config) WAV
# Output: separated sources (speech + accompaniment)
```

**Test Conditions:**
| Condition | Priority | Expected |
|---|---|---|
| EN-M1 (music background) | **Critical** | Best of all models on this |
| EN-M2 (music + speech) | **Critical** | Key test |
| EN-A1 (all combined) | High | Key test |
| EN-C1 (clean 2-spk) | Medium | May trade quality |
| HI-M1 (Hindi + music) | High | Unknown: never trained on Hindi |

**What we expect to learn:**
- **Most important:** Does band-split architecture provide meaningful gains over standard TCN/Transformer on music mixtures?
- Does music-awareness hurt speech-only separation quality?
- Is the architecture compatible with speaker conditioning (band-level injection)?

**Compute estimate:** ~30 min A100 for all conditions.

---

## MODEL 5: AudioSep (Text-Conditioned Alternative)

### Setup
```bash
git clone https://github.com/Audio-AGI/AudioSep
pip install -r requirements.txt
# Model: AudioSep-base (LAION-CLAP + ResUNet)

# Input: mixture WAV + text query
# Example queries:
#   "a person speaking in English"
#   "a male voice speaking Hindi"
#   "a woman's voice in conversation"
```

**Note:** AudioSep uses natural language class descriptions, not speaker-specific conditioning. We'll test with both generic and increasingly specific queries:
- Query A: "a person speaking"
- Query B: "a male/female voice speaking" (using ground-truth gender)
- Query C: "a voice speaking [language]" (using ground-truth language)

**Test Conditions:**
| Condition | Priority | Expected |
|---|---|---|
| EN-C1 (clean 2-spk) | High | Moderate: can't distinguish speakers |
| EN-M1 (music background) | High | Good: text separates speech from music |
| EN-A1 (all combined) | Medium | Good for speech/non-speech; bad for speaker ID |
| HI-C1 (Hindi) | High | Unknown: CLAP is multilingual? |
| CS-C1 (code-switched) | Medium | Interesting: text queries in Hindi |

**What we expect to learn:**
- Whether text conditioning (without voice) can substitute for speaker conditioning
- How much speaker-specific information is needed beyond language/gender
- Whether CLAP-based conditioning handles Hindi/multilingual queries
- Upper bound for "speaker-agnostic voice extraction" vs. "speaker-specific extraction"

**Compute estimate:** ~40 min A100 for all conditions (CLAP encoding overhead).

---

## ABLATION ANALYSIS FRAMEWORK

### Dimension 1: Conditioning Type
| Model | Conditioning | Effect |
|---|---|---|
| SepFormer | None (blind) | Permutation ambiguity |
| SpEx+ (real enrollment) | Speaker embedding (real) | Best possible speaker-cond |
| SpEx+ (TTS enrollment) | Speaker embedding (synth) | Our deployment scenario |
| AudioSep | Text query | No speaker identity |
| Our model (proposed) | Embedding + T-F template | Hypothesis: best |

### Dimension 2: Backbone Architecture
| Model | Backbone | Music | Noise | Streaming |
|---|---|---|---|---|
| SpEx+ | TCN | Weak | Medium | No |
| SepFormer | Transformer | Weak | Medium | No |
| TF-GridNet | T-F LSTM | Medium | Strong | No |
| BSRNN | Band-split RNN | Strong | Medium | Partial |
| Mamba-based | SSM | TBD | TBD | Yes |

### Dimension 3: Language Generalization
| Model | English | Hindi | Code-switch |
|---|---|---|---|
| SpEx+ | ✓ | ✗ | ✗ |
| SepFormer | ✓ | ~✓ | ~✓ |
| TF-GridNet | ✓ | ~✓ | ~✓ |
| BSRNN | ✓ | ? | ? |
| AudioSep | ✓ | ? | ? |

### Key Hypotheses to Test
1. **H1:** TTS enrollment causes < 2 dB SI-SDR degradation vs. real enrollment in clean English.
2. **H2:** All models degrade > 5 dB on Hindi vs. English for speaker-conditioned models.
3. **H3:** BSRNN provides > 3 dB SI-SDR gain on music-mixed conditions vs. TCN-based models.
4. **H4:** Blind separation (SepFormer) beats speaker-conditioned SpEx+ on Hindi (due to language mismatch in SpEx+ speaker encoder).
5. **H5:** T-F domain (TF-GridNet) handles reverberant conditions better than time-domain models.

---

## EVALUATION SCRIPTS

### Unified evaluation runner
```python
# evaluate_model.py (pseudocode)
def evaluate_model(model_name, model_fn, test_conditions, enrollment_type):
    results = {}
    for condition in test_conditions:
        samples = load_condition(condition)
        for sample in samples:
            mixture = load_wav(sample.mixture_file)
            reference = load_wav(sample.reference_file)
            
            if enrollment_type == 'real':
                enrollment = load_wav(sample.real_enrollment)
            elif enrollment_type == 'tts_xtts':
                enrollment = load_wav(sample.tts_enrollment['xtts_v2'])
            
            extracted = model_fn(mixture, enrollment)  # or model_fn(mixture) for blind
            
            # Metrics
            si_sdr = compute_si_sdr(extracted, reference)
            pesq = compute_pesq(extracted, reference, sr=16000)
            stoi = compute_stoi(extracted, reference, sr=16000)
            wer = compute_wer(extracted, sample.transcript)
            dnsmos = compute_dnsmos(extracted)
            spk_sim = compute_speaker_similarity(extracted, reference)
            
            results[sample.id] = {
                'si_sdr': si_sdr, 'pesq': pesq, 'stoi': stoi,
                'wer': wer, 'dnsmos': dnsmos, 'spk_sim': spk_sim
            }
    
    return pd.DataFrame(results).T
```

---

## TIMELINE

| Week | Task |
|---|---|
| Week 1 | Build test set (mixture generation + TTS enrollment synthesis) |
| Week 2 | Run SpEx+ ablation (real + TTS enrollment, all conditions) |
| Week 3 | Run SepFormer + TF-GridNet ablations |
| Week 4 | Run BSRNN + AudioSep ablations |
| Week 5 | Analyze results, validate hypotheses, inform architecture decisions |

**Total compute needed:** ~10 A100-hours for all 5 models across all conditions.
