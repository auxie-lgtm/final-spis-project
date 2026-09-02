import os
import shutil
from pathlib import Path

def collect_wav_files(source_dir, dest_dir, move=False):
    """
    Search source_dir (recursively) for .wav files and copy (or move)
    them into dest_dir.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for wav_file in source_path.rglob("*.wav"):
        if wav_file.is_file():
            target = dest_path / wav_file.name

            # Avoid overwriting files with the same name
            if target.exists():
                base, ext = target.stem, target.suffix
                i = 1
                while target.exists():
                    target = dest_path / f"{base}_{i}{ext}"
                    i += 1

            if move:
                shutil.move(str(wav_file), str(target))
            else:
                shutil.copy2(str(wav_file), str(target))
            count += 1

    print(f"Done. {count} .wav file(s) {'moved' if move else 'copied'} to {dest_path}")

if __name__ == "__main__":
    source = input("Source folder: ").strip()
    dest = input("Destination folder: ").strip()
    move_choice = input("Move instead of copy? (y/N): ").strip().lower() == "y"

    collect_wav_files(source, dest, move=move_choice)
