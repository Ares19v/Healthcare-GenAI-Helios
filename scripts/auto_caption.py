"""
auto_caption.py
---------------
Scans the dataset/ directory and auto-generates .txt caption stub files
for every image that doesn't already have one.

The trigger word is derived automatically from the folder name:
  20_HeliosSurgeon  →  "HeliosSurgeon"
  20_HeliosClinic   →  "HeliosClinic"

Behaviour:
  - Image has NO .txt  → creates one with "TriggerWord, " as content
  - Image HAS .txt but trigger word is missing at the front → prepends it
  - Image HAS .txt and already starts with trigger word → skips (correct)

After running this script, use Kohya_ss's WD14 Captioner (Utilities tab)
to enrich each caption with fine-grained detail tags.

Usage:
    .venv\\Scripts\\python.exe scripts\\auto_caption.py
"""

import re
from pathlib import Path

from PIL import Image  # used below: Image.open() validates each file is a real image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def extract_trigger_word(folder_name: str) -> str:
    """Strip the leading 'NN_' prefix and return the trigger word."""
    match = re.match(r"^\d+_(.+)$", folder_name)
    if match:
        return match.group(1)
    return folder_name


def process_folder(folder: Path, trigger: str) -> dict:
    stats = {"created": 0, "prepended": 0, "skipped": 0, "errors": 0}

    image_files = [
        f for f in folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.name != ".gitkeep"
    ]

    if not image_files:
        print(f"  [WARN] No images found in {folder.name}")
        return stats

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")

        try:
            # Validate the file is actually a readable image
            with Image.open(img_path):
                pass
        except Exception as e:
            print(f"  [ERROR] Cannot open {img_path.name}: {e}")
            stats["errors"] += 1
            continue

        prefix = f"{trigger}, "

        if not txt_path.exists():
            txt_path.write_text(prefix, encoding="utf-8")
            print(f"  [CREATED]   {txt_path.name}")
            stats["created"] += 1

        else:
            existing = txt_path.read_text(encoding="utf-8").strip()
            if not existing.startswith(trigger):
                updated = prefix + existing
                txt_path.write_text(updated, encoding="utf-8")
                print(f"  [PREPENDED] {txt_path.name}")
                stats["prepended"] += 1
            else:
                print(f"  [OK]        {txt_path.name}")
                stats["skipped"] += 1

    return stats


def main():
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        return

    total = {"created": 0, "prepended": 0, "skipped": 0, "errors": 0}

    for folder in sorted(DATASET_DIR.iterdir()):
        if not folder.is_dir():
            continue

        trigger = extract_trigger_word(folder.name)
        print(f"\n[Folder] {folder.name}  →  trigger: '{trigger}'")
        stats = process_folder(folder, trigger)

        for key in total:
            total[key] += stats[key]

    print("\n" + "=" * 50)
    print(f"Done.  Created: {total['created']}  |  Prepended: {total['prepended']}"
          f"  |  Already OK: {total['skipped']}  |  Errors: {total['errors']}")
    print("Next step: Run Kohya_ss WD14 Captioner to add detail tags.")


if __name__ == "__main__":
    main()
