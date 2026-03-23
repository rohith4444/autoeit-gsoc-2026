"""
Unit tests for preprocess.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import AUDIO_CONFIG, RAW_DIR, TRIMMED_DIR, preprocess


class TestPreprocess(unittest.TestCase):

    def test_raw_audio_files_exist(self):
        """All 4 raw mp3 files should be present in data/raw/."""
        for audio_id, config in AUDIO_CONFIG.items():
            filepath = RAW_DIR / config["file"]
            self.assertTrue(filepath.exists(), f"Missing: {filepath}")

    def test_audio_config_skip_values(self):
        """038012 skips 12min, others skip ~2:30."""
        self.assertEqual(AUDIO_CONFIG["038012_EIT-2A"]["skip_ms"], 720_000)
        for audio_id in ["038010_EIT-2A", "038011_EIT-1A", "038015_EIT-1A"]:
            self.assertEqual(AUDIO_CONFIG[audio_id]["skip_ms"], 150_000)

    def test_preprocess_all_files(self):
        """Each file should produce a non-empty wav in data/trimmed/."""
        for audio_id in AUDIO_CONFIG:
            out_path = preprocess(audio_id)
            self.assertTrue(out_path.exists(), f"Not created: {out_path}")
            self.assertEqual(out_path.suffix, ".wav")
            self.assertEqual(out_path.parent, TRIMMED_DIR)
            self.assertGreater(out_path.stat().st_size, 0)

    def test_invalid_audio_id(self):
        """Invalid audio_id should raise KeyError."""
        with self.assertRaises(KeyError):
            preprocess("nonexistent_file")


if __name__ == "__main__":
    unittest.main()
