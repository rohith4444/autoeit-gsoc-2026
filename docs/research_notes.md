# Research Notes — AutoEIT GSoC 2026

## 1. CrisperWhisper Methodology

### What it is
CrisperWhisper (Wagner et al., 2024) is a fine-tuned variant of Whisper Large v2 designed
for verbatim speech transcription with accurate word-level timestamps. Unlike standard
Whisper, which cleans up output by removing disfluencies and normalizing grammar,
CrisperWhisper is trained to preserve filler words, false starts, hesitations, and pauses
exactly as spoken.

### How it works
- Modified tokenizer that strips spaces from BPE tokens and makes spaces standalone
  tokens, enabling the DTW algorithm to detect and timestamp pauses between words
- Filler tokens for "uh" and "um" are repurposed from rarely used vocabulary tokens
  to ensure canonical and consistent representation of filled pauses
- A set of suitable decoder attention heads are selected and their cross-attention
  scores are averaged to construct the DTW cost matrix for word-level alignment
- Pause heuristics split pause durations evenly between preceding and subsequent words,
  with a cap at 160ms to distinguish artifact pauses from genuine speech pauses
- WavLM-style noise augmentation using overlapping speech, Gaussian noise, FSDnoisy18k,
  and AudioSet to improve robustness against background noise and multiple speakers
- Noise-only samples with empty transcriptions introduced in 1% of training data
  to mitigate hallucinations

### Training details
- Fine-tuned from whisper-large-v2 checkpoint
- 6,000 training steps
- Batch size: 256
- Learning rate: 0.00005 with linear decay and 800-step warmup
- Approximately 2 epochs

### Training data
- AMI Meeting Corpus — approximately 29,000 meeting recording clips with canonical
  filler transcriptions (English spontaneous speech)
- PodcastFillers Corpus — approximately 105,000 samples after augmentation with
  varying context lengths around each filler event (English)
- CommonVoice14 English subset — cleaned to remove non-verbatim samples using
  a 3% CER threshold
- Noise datasets: FSDnoisy18k and AudioSet for noise robustness training
- **Language limitation: trained primarily on English spontaneous speech.
  No Spanish training data. Falls back to standard Whisper behavior on Spanish audio.**

### Key results
- WER on AMI: 9.72 (vs Whisper baseline 16.82) — 42% improvement
- WER on TED-LIUM: 3.26 (vs Whisper baseline 4.01)
- Near perfect filler word detection on PodcastFillers test set
- Superior noise robustness compared to WhisperX and WhisperT

### Known weakness
The selection of attention heads used for DTW alignment is arbitrary rather than
principled — the chosen heads may not be optimal across all audio conditions.
The authors acknowledge this as an open problem.

### Future direction stated by authors
The authors explicitly identified transferring verbatim transcription capabilities
to other languages as a near-future goal. This is exactly what AutoEIT GSoC 2026
aims to do for Spanish.

### Relevance to AutoEIT
CrisperWhisper demonstrates that Whisper can be fine-tuned for verbatim transcription
using spontaneous conversational speech data. The same methodology — retokenization,
DTW alignment, noise augmentation, and verbatim fine-tuning — will be adapted for
Spanish using Fisher Spanish and CALLHOME Spanish corpora. This is the core technical
approach for the GSoC project.

### Key finding
No production-ready verbatim ASR model exists for Spanish. This is the gap
AutoEIT aims to fill.

**Reference:** Wagner, L., Thallinger, B., & Zusag, M. (2024). CrisperWhisper: Accurate
timestamps on verbatim speech transcriptions. Interspeech 2024, 1265–1269.
https://arxiv.org/abs/2408.16589

---