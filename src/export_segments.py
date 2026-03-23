"""
Export Participant Segments for AutoEIT

Reads the speaker detection results (JSON) and exports each participant
response as an individual wav file for transcription.
"""

import sys
import json
from pathlib import Path
from pydub import AudioSegment

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRIMMED_DIR = BASE_DIR / "data" / "trimmed"
DETECTION_DIR = BASE_DIR / "output" / "speaker_detection"
SEGMENTS_DIR = BASE_DIR / "data" / "participant_segments"


def export_segments(audio_id: str) -> list:
    """
    Export detected participant segments as individual wav files.

    Args:
        audio_id: Key like "038010_EIT-2A"

    Returns:
        List of paths to exported wav files.
    """
    # Load detection results
    json_path = DETECTION_DIR / f"{audio_id}_segments.json"
    if not json_path.exists():
        print(f"ERROR: Detection results not found: {json_path}")
        print("Run detect_speaker.py first.")
        sys.exit(1)

    with open(json_path) as f:
        detection = json.load(f)

    segments = detection["segments"]
    threshold = detection.get("threshold", 0)

    print(f"\n{'=' * 60}")
    print(f"Exporting segments: {audio_id}")
    print(f"{'=' * 60}")
    print(f"  {len(segments)} segments detected")

    # Filter out low-score segments (noise/false detections)
    filtered = [s for s in segments if s.get("avg_score", 1) >= threshold]
    skipped = len(segments) - len(filtered)
    if skipped:
        print(f"  Filtered out {skipped} low-score segment(s) (below threshold {threshold:.3f})")
    segments = filtered
    print(f"  {len(segments)} segments to export")

    # Load trimmed audio
    audio_path = TRIMMED_DIR / f"{audio_id}.wav"
    if not audio_path.exists():
        print(f"ERROR: Trimmed audio not found: {audio_path}")
        sys.exit(1)

    audio = AudioSegment.from_wav(str(audio_path))
    print(f"  Audio loaded: {len(audio) / 1000:.1f}s")

    # Create output directory
    out_dir = SEGMENTS_DIR / audio_id.split("_")[0]  # e.g., "038010"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export each segment
    exported = []
    for i, seg in enumerate(segments):
        start_ms = int(seg["start_sec"] * 1000)
        end_ms = int(seg["end_sec"] * 1000)

        # Clamp to audio bounds
        start_ms = max(0, start_ms)
        end_ms = min(len(audio), end_ms)

        segment_audio = audio[start_ms:end_ms]

        filename = f"response_{i + 1:02d}.wav"
        filepath = out_dir / filename
        segment_audio.export(str(filepath), format="wav")

        duration = (end_ms - start_ms) / 1000
        print(f"  [{i + 1:2d}] {seg['start_sec']:6.1f}s - {seg['end_sec']:6.1f}s  "
              f"({duration:.1f}s) → {filename}")
        exported.append(str(filepath))

    print(f"\n  Exported {len(exported)} files to {out_dir}")
    return exported


if __name__ == "__main__":
    audio_id = sys.argv[1] if len(sys.argv) > 1 else "038010_EIT-2A"
    export_segments(audio_id)
