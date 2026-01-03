import os
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from PIL import Image
import threading
from .models import PredictionHistory



# Path to model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "detector", "ml", "fake_logo_model.h5")

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224


def delete_file_later(path, delay=5):
    def delete():
        if os.path.exists(path):
            os.remove(path)
    threading.Timer(delay, delete).start()

@login_required
def home(request):
    result = None
    image_url = None

    if request.method == "POST" and request.FILES.get("logo"):
        uploaded_logo = request.FILES["logo"]

        fs = FileSystemStorage()
        filename = fs.save(uploaded_logo.name, uploaded_logo)
        file_path = fs.path(filename)
        image_url = fs.url(filename)


        # Image preprocessing
        img = Image.open(file_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        # Prediction
        prediction = model.predict(img_array)[0][0]

        if prediction >= 0.45:
            result = "REAL"
        else:
            result = "FAKE"

        # ✅ Save prediction history
        PredictionHistory.objects.create(
            user=request.user,
            image_name=filename,
            result=result,
            score=float(prediction)
        )

        # UI-friendly result with emoji
        result = f"{result} LOGO {'✅' if result == 'REAL' else '❌'}"

        

        delete_file_later(file_path, delay=5)
   

        # Cleanup
        # fs.delete(filename)
        

    return render(
    request,
    "detector/index.html",
    {
        "result": result,
        "image_url": image_url
    }
)


