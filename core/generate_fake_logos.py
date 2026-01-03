import cv2
import os
import numpy as np

REAL_DIR = "dataset/train/real"
FAKE_DIR = "dataset/train/fake"

os.makedirs(FAKE_DIR, exist_ok=True)

def add_noise(img):
    noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def rotate(img):
    h, w = img.shape[:2]
    angle = np.random.randint(-20, 20)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def flip(img):
    return cv2.flip(img, 1)

print("🔁 Starting fake logo generation...")

for file in os.listdir(REAL_DIR):
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    real_path = os.path.join(REAL_DIR, file)
    img = cv2.imread(real_path)

    if img is None:
        continue

    img = cv2.resize(img, (224, 224))
    name = os.path.splitext(file)[0]

    fake_variants = {
        f"{name}_noise.jpg": add_noise(img),
        f"{name}_blur.jpg": blur(img),
        f"{name}_rotate.jpg": rotate(img),
        f"{name}_flip.jpg": flip(img),
    }

    for fname, fimg in fake_variants.items():
        cv2.imwrite(os.path.join(FAKE_DIR, fname), fimg)

print("✅ Fake logo generation completed successfully.")
