"""
02_collect_students.py
-----------------------
Collect face images for each student.

Run once per student. Saves color face crops to:
    dataset/students/<name>/0000.jpg ...

Usage:
    python 02_collect_students.py
    python 02_collect_students.py --samples 80 --source 0

Tips for best results:
  - Ask student to slowly turn head left/right/up/down
  - Vary lighting if possible
  - 50-100 images per student is enough (transfer learning handles the rest)
"""

import os
import cv2
import time
import argparse

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
IMG_SIZE = 160  # matches training size


def collect(name: str, samples: int, source, dataset_dir: str):
    save_dir = os.path.join(dataset_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])

    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {source}")

    print(f"\n[INFO] Collecting {samples} images for '{name}'")
    print("[INFO] Move head slowly — left, right, up, down")
    print("[INFO] Press Q to stop early\n")

    count      = 0
    last_saved = 0

    while count < samples:
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            if time.time() - last_saved >= 0.12:
                crop = frame[y:y+h, x:x+w]
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                path = os.path.join(save_dir, f"{existing + count:04d}.jpg")
                cv2.imwrite(path, crop)
                count     += 1
                last_saved = time.time()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break  # one face per frame

        # Progress bar
        progress = int((count / samples) * 30)
        bar      = "█" * progress + "░" * (30 - progress)
        cv2.putText(frame, f"{name}: [{bar}] {count}/{samples}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Collecting Student Faces - Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] ✓ Saved {count} images for '{name}' → {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",     type=int, default=80)
    parser.add_argument("--source",      default="0")
    parser.add_argument("--dataset_dir", default="dataset/students")
    args = parser.parse_args()

    name = input("Enter student name (or roll number): ").strip()
    if not name:
        raise ValueError("Name cannot be empty!")

    collect(name, args.samples, args.source, args.dataset_dir)
