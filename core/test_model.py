import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "fake_logo_model.h5"
IMG_SIZE = 224

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    prediction = model.predict(arr)[0][0]
    return prediction

# 🔁 CHANGE THIS PATH TO TEST DIFFERENT IMAGES
TEST_IMAGE = "dataset/train/fake/adidas_01_blur.jpg"

# Example fake test:
# TEST_IMAGE = "dataset/train/fake/adidas_01_blur.jpg"

score = predict_image(TEST_IMAGE)

print("Raw model output:", score)

if score >= 0.7:
    print("Prediction: REAL LOGO ✅")
elif score <= 0.3:
    print("Prediction: FAKE LOGO ❌")
else:
    print("Prediction: UNCERTAIN ⚠️")
