"""
Cloud API Transcription for AutoEIT

Transcribes individual participant response segments using cloud STT APIs.
Supports AssemblyAI and OpenAI Whisper.
"""

import sys
import json
import time
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = BASE_DIR / "data" / "participant_segments"
OUTPUT_DIR = BASE_DIR / "output" / "transcriptions"


def ts():
    """Current timestamp for logging."""
    return time.strftime('%H:%M:%S')


# ---------------------------------------------------------------------------
# AssemblyAI
# ---------------------------------------------------------------------------

def transcribe_assemblyai_segment(audio_path: str) -> str:
    """Transcribe a single audio segment using AssemblyAI."""
    import assemblyai as aai

    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not aai.settings.api_key:
        print("ERROR: ASSEMBLYAI_API_KEY not set in .env")
        sys.exit(1)

    config = aai.TranscriptionConfig(language_code="es")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        return f"[ERROR: {transcript.error}]"

    return transcript.text.strip() if transcript.text else "[no response]"


# ---------------------------------------------------------------------------
# AssemblyAI with disfluencies
# ---------------------------------------------------------------------------

def transcribe_assemblyai_disfluencies_segment(audio_path: str) -> str:
    """Transcribe using AssemblyAI with disfluencies=true and verbatim prompt."""
    import assemblyai as aai

    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not aai.settings.api_key:
        print("ERROR: ASSEMBLYAI_API_KEY not set in .env")
        sys.exit(1)

    config = aai.TranscriptionConfig(
        language_detection=True,
        disfluencies=True,
        speech_models=["universal-3-pro", "universal-2"],
        prompt="Verbatim transcript of a non-native Spanish speaker. Include all filler words, hesitations, repetitions, and false starts exactly as spoken in Spanish, including 'eh', 'este', 'bueno', 'o sea', 'pues', 'mmm', 'um'. Do not correct any grammar or vocabulary errors. Transcribe exactly what was said.",
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        return f"[ERROR: {transcript.error}]"

    return transcript.text.strip() if transcript.text else "[no response]"


# ---------------------------------------------------------------------------
# OpenAI Whisper
# ---------------------------------------------------------------------------

def transcribe_openai_segment(audio_path: str) -> str:
    """Transcribe a single audio segment using OpenAI Whisper API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="es",
        )

    return response.text.strip() if response.text else "[no response]"


# ---------------------------------------------------------------------------
# Main transcription logic
# ---------------------------------------------------------------------------

PROVIDERS = {
    "assemblyai": transcribe_assemblyai_segment,
    "assemblyai_disfluencies": transcribe_assemblyai_disfluencies_segment,
    "openai": transcribe_openai_segment,
}


def transcribe_segments(audio_id: str, provider: str) -> list:
    """
    Transcribe all participant segments for one audio file using a cloud API.

    Args:
        audio_id: Key like "038010_EIT-2A"
        provider: "assemblyai" or "openai"

    Returns:
        List of dicts with sentence_num and text.
    """
    if provider not in PROVIDERS:
        print(f"Unknown provider: {provider}")
        print(f"Available: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    transcribe_fn = PROVIDERS[provider]

    # Find segment files
    participant_id = audio_id.split("_")[0]
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
    print(f"[{ts()}] Transcribing: {audio_id} with {provider}")
    print(f"{'=' * 60}")
    print(f"  [{ts()}] Found {len(segment_files)} segments")

    results = []
    total_start = time.time()

    for seg_file in segment_files:
        sentence_num = int(seg_file.stem.split("_")[1])

        start_time = time.time()
        print(f"  [{ts()}] Segment {sentence_num} ({seg_file.name})...", end=" ", flush=True)

        try:
            text = transcribe_fn(str(seg_file))
            elapsed = time.time() - start_time

            entry = {
                "sentence_num": sentence_num,
                "text": text,
                "source_file": seg_file.name,
                "provider": provider,
            }
            results.append(entry)
            print(f"({elapsed:.1f}s) [{sentence_num:2d}] {text}")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"({elapsed:.1f}s) [{sentence_num:2d}] ERROR: {str(e)[:100]}")
            entry = {
                "sentence_num": sentence_num,
                "text": "[TRANSCRIPTION FAILED]",
                "source_file": seg_file.name,
                "provider": provider,
                "error": str(e)[:200],
            }
            results.append(entry)

    total_elapsed = time.time() - total_start
    print(f"\n  [{ts()}] Total time: {total_elapsed:.1f}s")

    # Save results
    out_dir = OUTPUT_DIR / provider
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{audio_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [{ts()}] Saved to {out_path}")

    return results


if __name__ == "__main__":
    # Usage:
    #   python transcribe_cloud.py assemblyai 038010_EIT-2A
    #   python transcribe_cloud.py openai 038010_EIT-2A
    if len(sys.argv) < 2:
        print("Usage: python transcribe_cloud.py <provider> [audio_id]")
        print(f"Providers: {', '.join(PROVIDERS.keys())}")
        sys.exit(1)

    provider = sys.argv[1]
    audio_id = sys.argv[2] if len(sys.argv) > 2 else "038010_EIT-2A"
    transcribe_segments(audio_id, provider)
