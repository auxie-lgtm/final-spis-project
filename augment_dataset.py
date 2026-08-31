from __future__ import annotations

import shutil
from pathlib import Path
import os

import librosa
import numpy as np
import soundfile as sf

class AugmentDataset:
    ROOT = Path(__file__).resolve().parent / "sight-singing-vocal-data" / "dataset"
    OUTPUT_ROOT = Path(__file__).resolve().parent / "sight-singing-vocal-data" / "dataset_augmented"

    def get_augmented_dataset(self):
        if self.OUTPUT_ROOT.exists():
            return self.OUTPUT_ROOT
        return self.build_augmented_dataset()

    def __find_audio_path(self, sample_dir: Path) -> Path | None:
        for ext in (".mp3", ".wav", ".flac"):
            path = sample_dir / f"{sample_dir.name}{ext}"
            if path.exists():
                return path
        for path in sorted(sample_dir.iterdir()):
            if path.suffix.lower() in {".mp3", ".wav", ".flac"}:
                return path
        return None


    def __copy_label(self, sample_dir: Path, target_dir: Path) -> None:
        label_candidates = [
            sample_dir / f"{sample_dir.name}_label.txt",
            sample_dir / f"{sample_dir.name}label.txt",
        ]
        for label_path in label_candidates:
            if label_path.exists():
                shutil.copy2(label_path, target_dir / f"{target_dir.name}_label.txt")
                return
        for path in sorted(sample_dir.iterdir()):
            if path.name.endswith("_label.txt") or path.name.endswith("label.txt"):
                shutil.copy2(path, target_dir / f"{target_dir.name}_label.txt")
                return


    def __add_noise(self, y: np.ndarray, amount: float = 0.01) -> np.ndarray:
        noise = np.random.randn(len(y)) * amount
        return np.clip(y + noise, -1.0, 1.0)


    def __save_audio(self, target_dir: Path, filename: str, y: np.ndarray, sr: int) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / filename
        sf.write(output_path, y.astype(np.float32), sr)


    def __generate_variants_for_sample(self, sample_dir: Path, target_root: Path) -> int:
        audio_path = self.__find_audio_path(sample_dir)
        if audio_path is None:
            return 0

        y, sr = sf.read(str(audio_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

        variants = [
            ("orig", lambda x: x),
            ("pitch_up", lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=2)),
            ("pitch_down", lambda x: librosa.effects.pitch_shift(x, sr=sr, n_steps=-2)),
            ("time_fast", lambda x: librosa.effects.time_stretch(x, rate=1.08)),
            ("time_slow", lambda x: librosa.effects.time_stretch(x, rate=0.92)),
            ("gain_up", lambda x: np.clip(x * 1.25, -1.0, 1.0)),
            ("noise", lambda x: self.__add_noise(x, 0.02)),
        ]

        created = 0
        for name, transform in variants:
            new_dir = target_root / f"{sample_dir.name}_{name}"
            new_y = transform(y)
            if len(new_y) == 0:
                continue
            self.__save_audio(new_dir, f"{new_dir.name}.mp3", new_y, sr)
            self.__copy_label(sample_dir, new_dir)
            created += 1
        return created


    def build_augmented_dataset(self, source_root: Path = ROOT, out_root: Path = OUTPUT_ROOT, max_samples: int | None = None) -> int:
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        sample_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()])
        if max_samples is not None:
            sample_dirs = sample_dirs[:max_samples]

        total = 0
        for sample_dir in sample_dirs:
            total += self.__generate_variants_for_sample(sample_dir, out_root)

        print(f"Generated {total} augmented samples in {out_root}")
        return total


def get_augmented_dataset():
    return AugmentDataset().get_augmented_dataset()

