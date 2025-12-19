import cv2
import os

# Paths
CASCADE_PATH = "scripts/haarcascade_frontalface_default.xml"
RAW_REAL = "data/raw_frames/real"
RAW_FAKE = "data/raw_frames/fake"
OUT_REAL = "data/faces/real"
OUT_FAKE = "data/faces/fake"

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_FAKE, exist_ok=True)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def crop_faces(input_dir, output_dir):
    images = os.listdir(input_dir)
    saved = 0
    skipped = 0

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            skipped += 1
            continue

        # Take the largest detected face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))

        cv2.imwrite(os.path.join(output_dir, img_name), face)
        saved += 1

    print(f"{output_dir} -> saved: {saved}, skipped: {skipped}")

crop_faces(RAW_REAL, OUT_REAL)
crop_faces(RAW_FAKE, OUT_FAKE)
