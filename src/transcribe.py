"""
Transcription for AutoEIT using CrisperWhisper

Transcribes individual participant response segments using CrisperWhisper,
a Whisper variant fine-tuned for verbatim transcription that preserves
disfluencies, false starts, filler words, and errors.
"""

import sys
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = BASE_DIR / "data" / "participant_segments"
OUTPUT_DIR = BASE_DIR / "output" / "transcriptions" / "crisper_whisper"

MODEL_ID = "nyrahealth/CrisperWhisper"


def ts():
    """Current timestamp for logging."""
    return time.strftime('%H:%M:%S')


def load_pipeline():
    """Load the CrisperWhisper model and return a transcription pipeline."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float32

    print(f"[{ts()}] Loading CrisperWhisper model...")
    print(f"[{ts()}] Device: {device}, dtype: {torch_dtype}")

    print(f"[{ts()}] Loading model weights...", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    print(f"[{ts()}] Model weights loaded, moving to {device}...", flush=True)
    model.to(device)
    print(f"[{ts()}] Model on {device}", flush=True)

    print(f"[{ts()}] Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[{ts()}] Processor loaded", flush=True)

    print(f"[{ts()}] Creating pipeline...", flush=True)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )
    print(f"[{ts()}] Pipeline created", flush=True)

    return pipe


def transcribe_segments(audio_id: str, pipe=None) -> list:
    """
    Transcribe all participant segments for one audio file.

    Args:
        audio_id: Key like "038010_EIT-2A"
        pipe: Pre-loaded pipeline (optional, loads if None)

    Returns:
        List of dicts with sentence_num and text.
    """
    # Find segment files
    participant_id = audio_id.split("_")[0]  # e.g., "038010"
    seg_dir = SEGMENTS_DIR / participant_id
    if not seg_dir.exists():
        print(f"ERROR: Segments not found: {seg_dir}")
        print("Run export_segments.py first.")
        sys.exit(1)

    segment_files = sorted(seg_dir.glob("response_*.wav"))
    if not segment_files:
        print(f"ERROR: No response wav files in {seg_dir}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Transcribing: {audio_id}")
    print(f"{'=' * 60}")
    print(f"  [{ts()}] Found {len(segment_files)} segments")

    # Load model if not provided
    if pipe is None:
        pipe = load_pipeline()

    # Transcribe each segment
    results = []
    flagged = []
    total_start = time.time()

    for seg_file in segment_files:
        sentence_num = int(seg_file.stem.split("_")[1])  # response_01.wav → 1

        start_time = time.time()
        print(f"  [{ts()}] Segment {sentence_num} ({seg_file.name})...", end=" ", flush=True)

        try:
            result = pipe(
                str(seg_file),
                generate_kwargs={"language": "es"},
            )

            elapsed = time.time() - start_time
            text = result["text"].strip()

            entry = {
                "sentence_num": sentence_num,
                "text": text,
                "source_file": seg_file.name,
                "flagged": False,
            }
            results.append(entry)
            print(f"({elapsed:.1f}s) [{sentence_num:2d}] {text}")

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            print(f"({elapsed:.1f}s) [{sentence_num:2d}] ⚠ ERROR: {error_msg[:100]}")
            flagged.append(sentence_num)

            # Retry with a no-timestamps pipeline
            try:
                print(f"  [{ts()}] Retrying segment {sentence_num} without timestamps...", end=" ", flush=True)
                retry_start = time.time()

                # Create a simple no-timestamps pipeline on first retry
                if not hasattr(transcribe_segments, '_retry_pipe'):
                    transcribe_segments._retry_pipe = pipeline(
                        "automatic-speech-recognition",
                        model=pipe.model,
                        tokenizer=pipe.tokenizer,
                        feature_extractor=pipe.feature_extractor,
                        chunk_length_s=30,
                        batch_size=1,
                        return_timestamps=False,
                        torch_dtype=torch.float32,
                        device=pipe.device,
                    )

                retry_result = transcribe_segments._retry_pipe(
                    str(seg_file),
                    generate_kwargs={"language": "es"},
                )
                text = retry_result["text"].strip()

                retry_elapsed = time.time() - retry_start
                entry = {
                    "sentence_num": sentence_num,
                    "text": text,
                    "source_file": seg_file.name,
                    "flagged": True,
                    "flag_reason": "timestamp_error_retry_without_timestamps",
                }
                results.append(entry)
                print(f"({retry_elapsed:.1f}s) [{sentence_num:2d}] {text} [RETRIED - no timestamps]")

            except Exception as e2:
                print(f"  [{ts()}] Retry also failed: {str(e2)[:100]}")
                entry = {
                    "sentence_num": sentence_num,
                    "text": "[TRANSCRIPTION FAILED]",
                    "source_file": seg_file.name,
                    "flagged": True,
                    "flag_reason": f"both_attempts_failed: {error_msg[:80]}",
                }
                results.append(entry)

    total_elapsed = time.time() - total_start
    print(f"\n  [{ts()}] Total transcription time: {total_elapsed:.1f}s")

    if flagged:
        print(f"  ⚠ Flagged segments (timestamp errors): {flagged}")
    else:
        print(f"  All {len(results)} segments transcribed successfully")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{audio_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [{ts()}] Saved to {out_path}")

    return results


if __name__ == "__main__":
    audio_id = sys.argv[1] if len(sys.argv) > 1 else "038010_EIT-2A"
    transcribe_segments(audio_id)
