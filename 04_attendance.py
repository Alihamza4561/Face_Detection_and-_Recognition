import os
import csv
import json
import pickle
import argparse
from datetime import datetime
from collections import Counter, defaultdict

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE         = 160
CNN_THRESHOLD    = 0.75   
EMBED_THRESHOLD  = 0.55   
SMOOTH_FRAMES    = 8   

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def mark_attendance(name: str, attendance_dir: str = "attendance") -> bool:
    os.makedirs(attendance_dir, exist_ok=True)
    today     = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(attendance_dir, f"{today}.csv")

    already_marked = set()
    if os.path.exists(file_path):
        with open(file_path, "r", newline="") as f:
            for row in csv.reader(f):
                if row:
                    already_marked.add(row[0])

    if name in already_marked or name == "Unknown":
        return False

    write_header = not os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Name", "Time", "Date"])
        writer.writerow([name, datetime.now().strftime("%H:%M:%S"), today])

    print(f"[✓ ATTENDANCE] {name} marked present at {datetime.now().strftime('%H:%M:%S')}")
    return True


def load_models():
    required = [
        "models/student_model.keras",
        "models/student_labels.json",
        "models/student_encodings.pkl",
    ]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run 03_finetune_students.py first!"
            )

    print("[INFO] Loading student model...")
    model = tf.keras.models.load_model("models/student_model.keras")

    # Encoder for embedding extraction
    encoder = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer("l2_norm").output,
    )

    with open("models/student_labels.json", "r") as f:
        label_names = json.load(f)

    with open("models/student_encodings.pkl", "rb") as f:
        enc_data = pickle.load(f)

    known_encodings = enc_data["encodings"]
    known_names     = enc_data["names"]

    print(f"[INFO] Loaded model for {len(label_names)} students")
    print(f"[INFO] Loaded {len(known_encodings)} face embeddings")

    return model, encoder, label_names, known_encodings, known_names


def predict_face(face_bgr, model, encoder, label_names, known_encodings, known_names):
    """
    Returns (name, confidence) using dual CNN + embedding check.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    inp      = face_rgb.astype("float32") / 255.0
    inp      = np.expand_dims(inp, axis=0)

    preds      = model.predict(inp, verbose=0)[0]
    cnn_idx    = int(np.argmax(preds))
    cnn_conf   = float(preds[cnn_idx])
    cnn_name   = label_names[cnn_idx] if cnn_conf >= CNN_THRESHOLD else "Unknown"

    embedding  = encoder.predict(inp, verbose=0)[0]
    distances  = [
        float(np.linalg.norm(embedding - known_enc))
        for known_enc in known_encodings
    ]
    best_dist  = min(distances)
    best_idx   = int(np.argmin(distances))
    embed_name = known_names[best_idx] if best_dist < EMBED_THRESHOLD else "Unknown"

    if cnn_name != "Unknown" and embed_name != "Unknown" and cnn_name == embed_name:
        return cnn_name, cnn_conf          
    elif cnn_name != "Unknown" and embed_name == "Unknown":
        return "Unknown", cnn_conf        
    elif cnn_name == embed_name == "Unknown":
        return "Unknown", 0.0
    else:
        return "Unknown", 0.0             


def run(args):
    model, encoder, label_names, known_encodings, known_names = load_models()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {source}")

    print(f"\n[INFO] Attendance system running. Press Q to quit.\n")

    marked_today    = set()
    pred_buffers    = defaultdict(list)   
    flash_messages  = []                 

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            name, conf = predict_face(
                face_crop, model, encoder,
                label_names, known_encodings, known_names
            )

            bucket = (x // 80, y // 80)
            pred_buffers[bucket].append(name)
            if len(pred_buffers[bucket]) > SMOOTH_FRAMES:
                pred_buffers[bucket].pop(0)

            smoothed = Counter(pred_buffers[bucket]).most_common(1)[0][0]

            if smoothed != "Unknown":
                color = (0, 220, 0)
            else:
                color = (0, 0, 220)

        
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            label   = f"{smoothed} ({conf*100:.0f}%)" if smoothed != "Unknown" else "Unknown"
            txt_sz  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            cv2.rectangle(frame, (x, y-30), (x + txt_sz[0] + 8, y), color, cv2.FILLED)
            cv2.putText(frame, label, (x+4, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            if smoothed != "Unknown" and smoothed not in marked_today:
                if mark_attendance(smoothed, args.attendance_dir):
                    marked_today.add(smoothed)
                    flash_messages.append((f"✓ {smoothed} — Attendance Marked!", 60))

        cv2.rectangle(frame, (0, 0), (320, 40), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, f"Present today: {len(marked_today)} students",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)

        y_offset = 80
        updated_flash = []
        for msg, frames_left in flash_messages:
            cv2.putText(frame, msg, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            y_offset += 35
            if frames_left > 1:
                updated_flash.append((msg, frames_left - 1))
        flash_messages = updated_flash

        cv2.imshow("Attendance System - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Session ended. {len(marked_today)} students marked present.")
    if marked_today:
        print(f"[INFO] Present: {', '.join(sorted(marked_today))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",         default="0")
    parser.add_argument("--attendance_dir", default="attendance")
    args = parser.parse_args()
    run(args)
