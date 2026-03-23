# AutoEIT — Audio-to-Text Transcription for Spanish EIT

GSoC 2026 Evaluation Test for the [AutoEIT project](https://humanai.foundation/gsoc/2026/proposal_AutoEIT1.html) under the [HumanAI Foundation](https://humanai.foundation/).

## Overview

This project builds an automated pipeline to transcribe audio recordings from the Spanish Elicited Imitation Task (EIT), a sentence-repetition test used to measure language proficiency. The pipeline processes raw audio files, isolates participant responses using speaker voiceprint detection, and generates verbatim transcriptions using multiple ASR models.

## Pipeline

1. **Preprocess** — Trim audio files to remove English instructions and practice sentences
2. **Speaker Detection** — Use SpeechBrain ECAPA-TDNN speaker embeddings to identify and extract participant speech from mixed single-channel recordings
3. **Segment Export** — Export 30 individual response clips per participant
4. **Transcription** — Transcribe using 3 ASR models: OpenAI Whisper, CrisperWhisper, and AssemblyAI (with disfluency detection)
5. **Evaluation** — Compute CER/WER/SER against known stimulus sentences
6. **Output** — Select best transcriptions and document in Excel

## Results

- **120 transcriptions** (30 sentences x 4 participants)
- **Primary model:** OpenAI Whisper (WER: 0.479, CER: 0.353, 18/120 exact matches)
- Final output: `data/AutoEIT Sample Audio for Transcribing Updated.xlsx`

## Setup

### Prerequisites
- Python 3.11+
- ffmpeg (`brew install ffmpeg` on macOS)
- API keys for OpenAI and AssemblyAI

### Installation

```bash
git clone https://github.com/yourusername/autoeit-gsoc-2026.git
cd autoeit-gsoc-2026

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Pipeline

Place the 4 audio files in `data/raw/` and run each step in order:

```bash
# Step 1: Preprocess (trim audio files)
python src/preprocess.py

# Step 2: Speaker detection (requires reference timestamps in detect_speaker.py)
python src/detect_speaker.py 038010_EIT-2A
python src/detect_speaker.py 038011_EIT-1A
python src/detect_speaker.py 038012_EIT-2A
python src/detect_speaker.py 038015_EIT-1A

# Step 3: Export participant segments
python src/export_segments.py 038010_EIT-2A
python src/export_segments.py 038011_EIT-1A
python src/export_segments.py 038012_EIT-2A
python src/export_segments.py 038015_EIT-1A

# Step 4: Transcribe with cloud APIs
python src/transcribe_cloud.py openai 038010_EIT-2A
python src/transcribe_cloud.py assemblyai_disfluencies 038010_EIT-2A
# Repeat for other files...

# Step 4b: Transcribe with CrisperWhisper (local)
python src/transcribe.py 038010_EIT-2A
# Repeat for other files...

# Step 5: Evaluate
python src/evaluate.py

# Step 6: Export to Excel
python src/export_excel.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
autoeit-gsoc-2026/
├── data/
│   ├── raw/                          # Original MP3 recordings
│   ├── trimmed/                      # Preprocessed WAV/MP3 files
│   ├── references/                   # Speaker reference samples
│   ├── participant_segments/         # 30 individual response clips per participant
│   └── *.xlsx                        # Original and updated Excel files
├── src/
│   ├── preprocess.py                 # Audio trimming
│   ├── detect_speaker.py             # SpeechBrain ECAPA-TDNN speaker detection
│   ├── export_segments.py            # Export individual response clips
│   ├── transcribe.py                 # CrisperWhisper transcription
│   ├── transcribe_cloud.py           # OpenAI Whisper + AssemblyAI transcription
│   ├── evaluate.py                   # CER/WER/SER evaluation
│   ├── export_excel.py               # Export transcriptions to Excel
│   └── analyze_tone.py               # Tone detection analysis (exploratory)
├── tests/
│   └── test_preprocess.py            # Unit tests
├── notebooks/
│   ├── autoeit_transcription.ipynb   # Full analysis notebook
│   └── autoeit_transcription.pdf     # PDF export of notebook
├── output/
│   ├── transcriptions/               # JSON outputs per model
│   ├── speaker_detection/            # Detection results and plots
│   └── evaluation/                   # Evaluation metrics
├── requirements.txt
├── .env.example
└── README.md
```

## Key Findings

1. **Speaker voiceprint detection** (ECAPA-TDNN) successfully segments participant responses from mixed single-channel recordings with ~98% accuracy, after six other approaches failed (amplitude-based segmentation, diarization, full-audio transcription, tone detection, audio fingerprinting, VAD).

2. **No Spanish verbatim ASR model exists.** CrisperWhisper preserves disfluencies only for English and German. This is a genuine gap in the field.

3. **OpenAI Whisper** provides the best overall transcription quality for this task, though it suppresses disfluencies and filler words.

4. **Evaluation without ground truth** is possible using stimulus-referenced scoring (validated by McGuire & Larson-Hall, 2025), but cannot fully separate ASR errors from participant errors.

## Technologies

- **SpeechBrain** — ECAPA-TDNN speaker embeddings for voiceprint detection
- **OpenAI Whisper API** — Primary ASR model
- **CrisperWhisper** — Verbatim ASR (English/German only)
- **AssemblyAI Universal-3 Pro** — ASR with disfluency detection
- **jiwer** — WER/CER computation
- **pydub** — Audio processing
- **scikit-learn** — GMM threshold tuning
