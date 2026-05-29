# ♻️ PilahYuk! — AI Waste Classifier

> Sistem klasifikasi sampah berbasis deep learning menggunakan MobileNetV2. Upload foto sampah, dapatkan prediksi kategori secara instan melalui REST API.

---

## ✅ Checklist Capstone

### Main Quest (Wajib)
- [✓] Membangun model Deep Learning menggunakan TensorFlow Functional API / Model Subclassing
- [✓] Mengimplementasikan komponen kustom lanjutan: **Custom Layer**, **Custom Loss Function**, **Custom Callback**
- [✓] Menyimpan dan mengekspor model dalam format `.keras` (TensorFlow siap produksi)
- [✓] Membuat kode sederhana untuk proses inference model

### Side Quest (Nilai Tambah)
- [✓] Mengembangkan REST API mandiri menggunakan **FastAPI** (`app.py`)
- [✓] Mengimplementasikan training dan evaluation loop kustom secara penuh dari awal menggunakan **tf.GradientTape**
- [✓] Mengintegrasikan **TensorBoard** untuk memantau dan memvisualisasikan metrik pelatihan
- [✓] Model memiliki performa baik — **Akurasi ≥ 85%** dan **MAE ≤ 0.02** ✓

---

## 📊 Hasil Training

| Metrik | Training | Validasi | Testing |
|--------|----------|----------|---------|
| Accuracy | 97.68% | 92.90% | **93.74%** |
| MAE | 0.0118 | 0.0199 | **0.0181** |
| Loss | 0.0820 | 0.2856 | 0.2300 |

---

## 🗂️ Kelas yang Didukung

| # | Kelas | # | Kelas |
|---|-------|---|-------|
| 1 | 👕 Clothes | 5 | 🔩 Logam |
| 2 | 🪟 Kaca | 6 | 🥬 Biological |
| 3 | 📦 Kardus | 7 | 🧴 Plastik |
| 4 | 📄 Kertas | 8 | 🗑️ Residu |

---

## 🧠 Arsitektur Model

```
Input Image (224×224)
    ↓
MobileNetV2 (pretrained ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense + Dropout
    ↓
Softmax → 8 Kelas
```

Transfer learning dengan custom metric MAE, custom callback untuk early stopping otomatis, dan class weighting untuk mengatasi data imbalance. Training loop dibangun dari awal menggunakan `tf.GradientTape`.

---

## ⚡ API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Prediksi jenis sampah dari gambar |

### Contoh Response `/predict`

```json
{
  "status": "success",
  "data": {
    "jenis_sampah": "Plastik",
    "confidence": 0.9731
  }
}
```

---

## 🚀 Langsung aja coba

```bash
# 1. Yuk Akses Langsung Modelnya
https://hugpy-klasifikasi-sampah-risol-matcha.hf.space/docs

# 2. Cek juga hasil websitenya
https://pilah-yuk.devlabfortirta.cloud
```

---

## 🛠️ Tech Stack

- **Model**: TensorFlow / Keras, MobileNetV2, scikit-learn
- **Training**: tf.GradientTape (custom loop), TensorBoard
- **API**: FastAPI, Uvicorn
- **Image Processing**: Pillow, NumPy
- **Visualisasi**: Matplotlib, Seaborn

---

## 📁 Struktur Project

```
pilayhuk/
├── app.py                   # FastAPI backend
├── main_capstone.ipynb      # Training notebook
├── sampah_classifier.keras  # Model (dihosting di Hugging Face)
├── requirements.txt
└── README.md
```

---
## 📌 Catatan

- Checklist Menggunakan API Generative AI untuk fitur tambahan atau fitur sekunder pada aplikasi diterapkan langsung pada backend Aplikasi dan bosa langsung di gunakan, tidak dicantumkan langsung pada Notebook yang dibuat
---

*Dibuat sebagai bagian dari Capstone Project — PilahYuk! 🌿*
