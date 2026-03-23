"""
Speaker Detection for AutoEIT using SpeechBrain ECAPA-TDNN

Takes a short reference sample of the participant's voice and slides their
voiceprint across the full trimmed audio to find all timestamps where the
participant is speaking. Outputs 30 participant response segments.
"""

import sys
import json
import torch
import torchaudio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRIMMED_DIR = BASE_DIR / "data" / "trimmed"
REF_DIR = BASE_DIR / "data" / "references"
OUTPUT_DIR = BASE_DIR / "output" / "speaker_detection"

SAMPLE_RATE = 16000
EXPECTED_SENTENCES = 30

# Reference timestamps (in the TRIMMED audio, in seconds)
# User identifies a clean 3-5 second spot where the participant is clearly speaking
REFERENCE_CONFIG = {
    "038010_EIT-2A": {"start_sec": 14, "end_sec": 18},
    "038011_EIT-1A": {"start_sec": 0, "end_sec": 3},
    "038012_EIT-2A": {"start_sec": 20, "end_sec": 23},
    "038015_EIT-1A": {"start_sec": 2, "end_sec": 4},
}


def load_audio(filepath: Path, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio file as mono tensor at target sample rate."""
    signal, fs = torchaudio.load(str(filepath))
    if fs != sr:
        signal = torchaudio.functional.resample(signal, fs, sr)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal


def extract_reference(audio_id: str) -> Path:
    """Extract the reference clip from the trimmed audio and save it."""
    config = REFERENCE_CONFIG[audio_id]
    audio_path = TRIMMED_DIR / f"{audio_id}.wav"

    signal = load_audio(audio_path)
    start_sample = int(config["start_sec"] * SAMPLE_RATE)
    end_sample = int(config["end_sec"] * SAMPLE_RATE)
    ref_signal = signal[:, start_sample:end_sample]

    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REF_DIR / f"{audio_id}_ref.wav"
    torchaudio.save(str(ref_path), ref_signal, SAMPLE_RATE)
    print(f"  Reference saved: {ref_path} ({config['end_sec'] - config['start_sec']}s)")
    return ref_path


def detect_participant(audio_id: str,
                       window_sec: float = 1.5,
                       hop_sec: float = 0.2,
                       merge_gap_sec: float = 1.5) -> list:
    """
    Detect participant speech using ECAPA-TDNN speaker embeddings.

    Args:
        audio_id: Key like "038010_EIT-2A"
        window_sec: Sliding window size in seconds for embedding extraction
        hop_sec: Hop between windows in seconds (0.2 = 5 scores per second)
        merge_gap_sec: Merge segments closer than this (handles mid-sentence pauses)

    Returns:
        List of dicts with start_sec, end_sec, duration_sec, score for each segment.
    """
    print(f"\n{'=' * 60}")
    print(f"Speaker Detection: {audio_id}")
    print(f"{'=' * 60}")

    # Step 1: Load ECAPA-TDNN model
    print("  Loading ECAPA-TDNN model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"),
    )
    cos_sim = torch.nn.CosineSimilarity(dim=-1)

    # Step 2: Extract and encode reference
    print("  Extracting reference sample...")
    ref_path = extract_reference(audio_id)
    ref_signal = load_audio(ref_path)
    ref_embedding = classifier.encode_batch(ref_signal).squeeze()
    print(f"  Reference embedding shape: {ref_embedding.shape}")

    # Step 3: Load full trimmed audio
    audio_path = TRIMMED_DIR / f"{audio_id}.wav"
    full_signal = load_audio(audio_path)
    total_duration = full_signal.shape[1] / SAMPLE_RATE
    print(f"  Full audio: {total_duration:.1f}s")

    # Step 4: Slide window and compute similarity
    print(f"  Sliding window: {window_sec}s window, {hop_sec}s hop...")
    window_samples = int(window_sec * SAMPLE_RATE)
    hop_samples = int(hop_sec * SAMPLE_RATE)

    timestamps = []
    scores = []

    for start in range(0, full_signal.shape[1] - window_samples, hop_samples):
        end = start + window_samples
        chunk = full_signal[:, start:end]

        # Normalize per chunk
        chunk = chunk / (chunk.abs().max() + 1e-8)

        chunk_embedding = classifier.encode_batch(chunk).squeeze()
        score = cos_sim(ref_embedding.unsqueeze(0), chunk_embedding.unsqueeze(0)).item()

        center_time = (start + end) / 2 / SAMPLE_RATE
        timestamps.append(center_time)
        scores.append(score)

    timestamps = np.array(timestamps)
    scores = np.array(scores)
    print(f"  Computed {len(scores)} similarity scores")
    print(f"  Score range: {scores.min():.3f} to {scores.max():.3f}, mean: {scores.mean():.3f}")

    # Step 5: Find optimal threshold using GMM
    from sklearn.mixture import GaussianMixture
    scores_reshaped = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(scores_reshaped)
    cluster_means = sorted(gmm.means_.flatten())
    threshold = np.mean(cluster_means)
    print(f"  GMM cluster centers: {cluster_means[0]:.3f} (non-participant), {cluster_means[1]:.3f} (participant)")
    print(f"  Auto threshold: {threshold:.3f}")

    # Step 6: Group consecutive above-threshold points into segments
    above = scores >= threshold
    segments = []
    in_segment = False
    seg_start = 0

    for i, (t, is_above) in enumerate(zip(timestamps, above)):
        if is_above and not in_segment:
            seg_start = t - window_sec / 2  # adjust to window start
            in_segment = True
        elif not is_above and in_segment:
            seg_end = timestamps[i - 1] + window_sec / 2  # adjust to window end
            segments.append({"start_sec": max(0, seg_start), "end_sec": seg_end})
            in_segment = False

    if in_segment:
        seg_end = timestamps[-1] + window_sec / 2
        segments.append({"start_sec": max(0, seg_start), "end_sec": min(total_duration, seg_end)})

    print(f"  Raw segments (before merge): {len(segments)}")

    # Step 7: Merge close segments (mid-sentence pauses)
    merged = []
    for seg in segments:
        if merged and seg["start_sec"] - merged[-1]["end_sec"] < merge_gap_sec:
            merged[-1]["end_sec"] = seg["end_sec"]
        else:
            merged.append(dict(seg))
    segments = merged

    # Add duration and average score
    for seg in segments:
        seg["duration_sec"] = round(seg["end_sec"] - seg["start_sec"], 2)
        # Average score for this segment's time range
        mask = (timestamps >= seg["start_sec"]) & (timestamps <= seg["end_sec"])
        seg["avg_score"] = round(float(np.mean(scores[mask])) if mask.any() else 0, 3)

    print(f"  After merging (gap < {merge_gap_sec}s): {len(segments)} segments")

    # Step 8: Validate
    if len(segments) != EXPECTED_SENTENCES:
        print(f"  WARNING: Expected {EXPECTED_SENTENCES}, got {len(segments)}")

    # Check for [no response] gaps
    if len(segments) >= 2:
        gaps = []
        for i in range(1, len(segments)):
            gap = segments[i]["start_sec"] - segments[i - 1]["end_sec"]
            if gap > 15:  # suspiciously long gap = possible missing response
                gaps.append((i, gap))
        if gaps:
            print(f"  WARNING: Large gaps detected (possible missing responses):")
            for idx, gap in gaps:
                print(f"    Between segment {idx} and {idx + 1}: {gap:.1f}s gap")

    # Print segments
    print(f"\n  Detected participant segments:")
    for i, seg in enumerate(segments):
        print(f"    [{i + 1:2d}] {seg['start_sec']:6.1f}s - {seg['end_sec']:6.1f}s  "
              f"({seg['duration_sec']:.1f}s)  score: {seg['avg_score']:.3f}")

    # Step 9: Save results and plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output = {
        "audio_id": audio_id,
        "threshold": round(threshold, 3),
        "total_segments": len(segments),
        "segments": segments,
    }
    json_path = OUTPUT_DIR / f"{audio_id}_segments.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved segments to {json_path}")

    # Plot similarity scores
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Top: similarity timeline
    axes[0].plot(timestamps, scores, linewidth=0.8, color='blue', alpha=0.7)
    axes[0].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
    axes[0].fill_between(timestamps, threshold, scores,
                          where=scores >= threshold, alpha=0.3, color='green', label='Participant')
    axes[0].set_title(f"Speaker Similarity: {audio_id}")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: score histogram
    axes[1].hist(scores, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
    axes[1].axvline(x=cluster_means[0], color='orange', linestyle=':', label=f'Cluster 1: {cluster_means[0]:.3f}')
    axes[1].axvline(x=cluster_means[1], color='green', linestyle=':', label=f'Cluster 2: {cluster_means[1]:.3f}')
    axes[1].set_title("Score Distribution")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"{audio_id}_similarity.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot to {plot_path}")

    return segments


if __name__ == "__main__":
    audio_id = sys.argv[1] if len(sys.argv) > 1 else "038010_EIT-2A"

    if audio_id not in REFERENCE_CONFIG:
        print(f"No reference config for {audio_id}")
        print(f"Available: {', '.join(REFERENCE_CONFIG.keys())}")
        sys.exit(1)

    detect_participant(audio_id)
