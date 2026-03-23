"""
Analyze a section of EIT audio to detect if a tone/beep exists between
stimulus and response, even if it's too quiet to hear easily.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal import spectrogram
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_section(audio_id="038010_EIT-2A", start_sec=270, end_sec=280):
    """
    Analyze a 10-second section of audio for tone detection.
    Default: 4:30 to 4:40 (270s to 280s)
    """
    filepath = RAW_DIR / f"{audio_id}.mp3"
    print(f"Loading {filepath.name}...")
    audio = AudioSegment.from_mp3(str(filepath))

    # Extract section
    section = audio[start_sec * 1000:end_sec * 1000]

    # Convert to numpy array
    samples = np.array(section.get_array_of_samples(), dtype=np.float64)
    sr = section.frame_rate

    # If stereo, take one channel
    if section.channels == 2:
        samples = samples[::2]

    # Normalize
    samples = samples / (np.max(np.abs(samples)) + 1e-10)

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(samples) / sr:.2f}s")
    print(f"Analyzing {start_sec}s to {end_sec}s...")

    # --- Plot 1: Waveform ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    time = np.arange(len(samples)) / sr + start_sec
    axes[0].plot(time, samples, linewidth=0.3)
    axes[0].set_title(f"Waveform: {audio_id} ({start_sec}s - {end_sec}s)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Spectrogram (shows frequency content over time) ---
    f, t, Sxx = spectrogram(samples, fs=sr, nperseg=1024, noverlap=768)
    t = t + start_sec  # offset time axis
    axes[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    axes[1].set_title("Spectrogram (all frequencies)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylim(0, 8000)  # Focus on 0-8kHz

    # --- Plot 3: High-frequency energy (tones often have energy in specific bands) ---
    # Look at different frequency bands over time
    freq_bands = {
        "500-1500 Hz (speech)": (500, 1500),
        "1500-3000 Hz (upper speech)": (1500, 3000),
        "3000-5000 Hz (possible tone)": (3000, 5000),
        "5000-8000 Hz (high freq)": (5000, 8000),
    }

    for label, (f_low, f_high) in freq_bands.items():
        mask = (f >= f_low) & (f < f_high)
        band_energy = np.sum(Sxx[mask, :], axis=0)
        band_energy_db = 10 * np.log10(band_energy + 1e-10)
        axes[2].plot(t, band_energy_db, label=label, alpha=0.8)

    axes[2].set_title("Energy by frequency band over time")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Energy (dB)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"tone_analysis_{audio_id}_{start_sec}s-{end_sec}s.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved analysis to {out_path}")

    # --- Also check spectral flatness for pure tone detection ---
    # Pure tones have very low spectral flatness
    print("\n--- Spectral Flatness Analysis ---")
    window_size = int(0.05 * sr)  # 50ms windows
    hop = int(0.025 * sr)  # 25ms hop

    flatness_values = []
    flatness_times = []

    for i in range(0, len(samples) - window_size, hop):
        window = samples[i:i + window_size]
        spectrum = np.abs(np.fft.rfft(window))
        spectrum = spectrum + 1e-10

        geo_mean = np.exp(np.mean(np.log(spectrum)))
        arith_mean = np.mean(spectrum)
        flatness = geo_mean / arith_mean

        flatness_values.append(flatness)
        flatness_times.append(i / sr + start_sec)

    flatness_values = np.array(flatness_values)

    # Find frames with very low flatness (potential tones) AND non-trivial energy
    energy_per_frame = []
    for i in range(0, len(samples) - window_size, hop):
        window = samples[i:i + window_size]
        energy_per_frame.append(np.sqrt(np.mean(window ** 2)))
    energy_per_frame = np.array(energy_per_frame)

    # Potential tones: low flatness + some energy
    energy_threshold = np.percentile(energy_per_frame, 30)
    tone_candidates = (flatness_values < 0.05) & (energy_per_frame > energy_threshold)

    if np.any(tone_candidates):
        candidate_times = np.array(flatness_times)[tone_candidates]
        print(f"Potential tone-like signals at: {candidate_times}s")
        # Group consecutive candidates
        groups = []
        current_group = [candidate_times[0]]
        for t_val in candidate_times[1:]:
            if t_val - current_group[-1] < 0.1:  # within 100ms
                current_group.append(t_val)
            else:
                groups.append(current_group)
                current_group = [t_val]
        groups.append(current_group)

        print(f"\nGrouped tone candidates:")
        for g in groups:
            print(f"  {g[0]:.2f}s - {g[-1]:.2f}s (duration: {g[-1]-g[0]:.3f}s)")
    else:
        print("No clear tone-like signals detected in this section.")


if __name__ == "__main__":
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 270
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 280
    analyze_section(start_sec=start, end_sec=end)
