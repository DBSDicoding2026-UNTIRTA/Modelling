from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # Tambahkan ini
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image

# 1. Inisialisasi app (HARUS DI SINI)
app = FastAPI()

# 2. Atur CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. Definisikan ulang custom metric MAE
def mae(y_true, y_pred):
    # Pastikan depth sesuai dengan jumlah kelas kamu (6)
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=6)
    return tf.reduce_mean(tf.abs(y_true_one_hot - y_pred))


# 4. Load Model
print("Memuat model AI PilahYuk!...")
# Pastikan file "sampah_classifier.keras" ada di folder yang sama dengan app.py
model = load_model("sampah_classifier.keras", custom_objects={"mae": mae})

class_names = ["Kaca", "Kardus", "Kertas", "Logam", "Plastik", "Residu"]


@app.get("/")
def read_root():
    return {"pesan": "API PilahYuk! Berjalan Lancar 🚀"}


@app.post("/predict")
async def predict_sampah(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))

        # Gunakan image.img_to_array dari Keras
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)

        label = class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        return {
            "status": "success",
            "data": {"jenis_sampah": label, "confidence": confidence},
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
