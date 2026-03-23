"""
Audio Preprocessing for AutoEIT

Trims EIT audio recordings to remove instructions and English practice,
then exports the relevant Spanish portion as wav files for STT processing.

No segmentation is done here — the full trimmed audio is passed to STT models,
which handle speaker separation via diarization.
"""

import sys
from pathlib import Path
from pydub import AudioSegment


# Configuration for each audio file
AUDIO_CONFIG = {
    "038010_EIT-2A": {"file": "038010_EIT-2A.mp3", "skip_ms": 150_000},  # skip ~2:30
    "038011_EIT-1A": {"file": "038011_EIT-1A.mp3", "skip_ms": 150_000},  # skip ~2:30
    "038012_EIT-2A": {"file": "038012_EIT-2A.mp3", "skip_ms": 720_000},  # skip 12:00
    "038015_EIT-1A": {"file": "038015_EIT-1A.mp3", "skip_ms": 146_000},  # skip 2:26
}

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
TRIMMED_DIR = BASE_DIR / "data" / "trimmed"


def preprocess(audio_id: str) -> Path:
    """
    Load an audio file, trim the non-relevant beginning, and export as wav.

    Args:
        audio_id: Key from AUDIO_CONFIG (e.g., "038010_EIT-2A")

    Returns:
        Path to the trimmed wav file
    """
    config = AUDIO_CONFIG[audio_id]
    filepath = RAW_DIR / config["file"]

    print(f"Loading {filepath.name}...")
    audio = AudioSegment.from_mp3(str(filepath))
    print(f"  Full duration: {len(audio) / 1000:.1f}s")

    # Trim the beginning (instructions + English practice)
    skip_ms = config["skip_ms"]
    trimmed = audio[skip_ms:]
    print(f"  Skipped first {skip_ms / 1000:.0f}s")
    print(f"  Trimmed duration: {len(trimmed) / 1000:.1f}s")

    # Export as both wav and mp3
    # wav = uncompressed, best quality for local models
    # mp3 = small file size, needed for cloud APIs with upload limits (e.g., OpenAI 25MB limit)
    TRIMMED_DIR.mkdir(parents=True, exist_ok=True)

    wav_path = TRIMMED_DIR / f"{audio_id}.wav"
    trimmed.export(str(wav_path), format="wav")
    print(f"  Saved wav: {wav_path}")

    mp3_path = TRIMMED_DIR / f"{audio_id}.mp3"
    trimmed.export(str(mp3_path), format="mp3")
    print(f"  Saved mp3: {mp3_path}")

    return wav_path


def preprocess_all() -> dict[str, Path]:
    """Preprocess all 4 audio files."""
    results = {}
    for audio_id in AUDIO_CONFIG:
        print(f"\n{'=' * 50}")
        results[audio_id] = preprocess(audio_id)

    print(f"\n{'=' * 50}")
    print("DONE — all files trimmed and saved to data/trimmed/")
    for audio_id, path in results.items():
        print(f"  {audio_id} → {path.name}")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_id = sys.argv[1]
        if audio_id not in AUDIO_CONFIG:
            print(f"Unknown audio ID: {audio_id}")
            print(f"Available: {', '.join(AUDIO_CONFIG.keys())}")
            sys.exit(1)
        preprocess(audio_id)
    else:
        preprocess_all()
