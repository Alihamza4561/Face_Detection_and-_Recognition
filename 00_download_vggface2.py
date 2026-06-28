"""
00_download_vggface2.py
------------------------
Downloads and prepares the VGGFace2 dataset for training.

VGGFace2 requires registration at:
https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/

Once you have downloaded the dataset manually, place the files as:
    vggface2_data/
        train/          <- extracted training set
        test/           <- extracted test set

Then run this script to prepare a subset for faster training:
    python 00_download_vggface2.py --people 500 --images_per_person 50

Why subset?
- Full VGGFace2 = 9,131 people, 3.3M images → needs days to train
- 500 people × 50 images = 25,000 images → trains in hours on CPU, minutes on GPU
- Still teaches the CNN rich, diverse face features
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def prepare_subset(
    vgg_train_dir: str,
    output_dir: str,
    num_people: int,
    images_per_person: int,
    seed: int = 42,
):
    random.seed(seed)

    vgg_path = Path(vgg_train_dir)
    out_path = Path(output_dir)

    if not vgg_path.exists():
        print(f"""
[ERROR] VGGFace2 train directory not found at: {vgg_train_dir}

To get VGGFace2:
1. Go to: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
2. Fill out the request form (academic/research use)
3. You will receive a download link via email
4. Download and extract so the structure is:
       vggface2_data/train/n000001/  (person folders)
       vggface2_data/train/n000002/
       ...
5. Re-run this script

Alternative faster option — MS-Celeb subset:
   pip install gdown
   gdown https://drive.google.com/uc?id=<shared-id>   (search for ms-celeb-1m subset on GitHub)
        """)
        return

    all_people = sorted([p for p in vgg_path.iterdir() if p.is_dir()])
    print(f"[INFO] Found {len(all_people)} people in VGGFace2 train set")

    selected_people = random.sample(all_people, min(num_people, len(all_people)))
    print(f"[INFO] Selecting {len(selected_people)} people, up to {images_per_person} images each")

    out_path.mkdir(parents=True, exist_ok=True)
    total_images = 0
    skipped_people = 0

    for person_dir in selected_people:
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        if len(images) < 10:
            skipped_people += 1
            continue

        selected_images = random.sample(images, min(images_per_person, len(images)))
        out_person = out_path / person_dir.name
        out_person.mkdir(exist_ok=True)

        for img in selected_images:
            shutil.copy(img, out_person / img.name)
            total_images += 1

    print(f"[INFO] Prepared {total_images} images across {len(selected_people) - skipped_people} people")
    print(f"[INFO] Saved to: {output_dir}")
    print(f"[INFO] Skipped {skipped_people} people with too few images")
    print(f"\n[NEXT] Run: python 01_train_vggface2.py --data_dir {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_train_dir",      default="vggface2_data/train",
                        help="Path to extracted VGGFace2 train folder")
    parser.add_argument("--output_dir",          default="dataset/vggface2_subset",
                        help="Where to save the prepared subset")
    parser.add_argument("--people",              type=int, default=500,
                        help="Number of people to include (default 500)")
    parser.add_argument("--images_per_person",   type=int, default=50,
                        help="Max images per person (default 50)")
    args = parser.parse_args()

    prepare_subset(
        args.vgg_train_dir,
        args.output_dir,
        args.people,
        args.images_per_person,
    )
