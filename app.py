import os
from flask import Flask, render_template, request, redirect, url_for
from inference_sdk import InferenceHTTPClient

# ==============================
# ROBOFLOW SETTINGS
# ==============================
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ.get("ROBOFLOW_API_KEY")  # ← dari env variable
)
MODEL_ID = "deteksi-penyakit-daun-tomat-sljaj/1"
MODEL_ACCURACY = 98.1

# ==============================
# LABEL INFO (Bahasa Indonesia)
# ==============================
LABEL_TRANSLATE = {
    "Tomato_Bacterial_spot": {
        "id": "Bercak Bakteri",
        "emoji": "🦠",
        "desc": "Daun tomat terkena bercak bakteri, biasanya muncul bercak coklat kehitaman di permukaan daun.",
        "tips": "Singkirkan daun terinfeksi, hindari penyiraman dari atas, gunakan pestisida bakterisida jika perlu.",
        "level": "danger"
    },
    "Tomato_Early_blight": {
        "id": "Hawar Dini",
        "emoji": "🍂",
        "desc": "Gejala berupa bercak coklat bulat di daun bawah, biasanya menyerang daun tua terlebih dahulu.",
        "tips": "Buang daun sakit, rotasi tanaman, gunakan fungisida bila parah.",
        "level": "warning"
    },
    "Tomato_Late_blight": {
        "id": "Hawar Daun Akhir",
        "emoji": "⚠️",
        "desc": "Daun dan buah jadi kecoklatan atau berair, sangat cepat menyebar ke seluruh tanaman.",
        "tips": "Segera buang bagian terinfeksi, hindari kelembaban tinggi, gunakan fungisida sistemik.",
        "level": "danger"
    },
    "Tomato_Leaf_Mold": {
        "id": "Jamur Daun",
        "emoji": "🍄",
        "desc": "Daun berubah kuning dan terdapat bercak jamur keabu-abuan di bagian bawah daun.",
        "tips": "Jaga ventilasi, jangan terlalu rapat menanam, semprot fungisida secara rutin.",
        "level": "warning"
    },
    "Tomato_Septoria_leaf_spot": {
        "id": "Bercak Septoria",
        "emoji": "🔵",
        "desc": "Bercak kecil bulat abu-abu dikelilingi warna coklat pada daun tua.",
        "tips": "Buang daun sakit, hindari daun basah terlalu lama, gunakan fungisida.",
        "level": "warning"
    },
    "Tomato_Spider_mites_Two-spotted_spider_mite": {
        "id": "Tungau Laba-laba",
        "emoji": "🕷️",
        "desc": "Daun tomat tampak kuning dan terdapat jaring halus di bawah daun akibat serangan tungau.",
        "tips": "Semprot air ke bawah daun, gunakan insektisida alami seperti neem oil.",
        "level": "warning"
    },
    "Tomato_Target_Spot": {
        "id": "Bercak Target",
        "emoji": "🎯",
        "desc": "Bercak bulat dengan pola lingkaran seperti target panah pada permukaan daun.",
        "tips": "Buang daun dan buah yang sakit, jaga kebersihan kebun.",
        "level": "warning"
    },
    "Tomato_Tomato_mosaic_virus": {
        "id": "Virus Mosaik",
        "emoji": "🧬",
        "desc": "Daun belang-belang kuning-hijau tidak merata, pertumbuhan tanaman melambat.",
        "tips": "Hancurkan tanaman yang sakit, jangan pegang tanaman sehat setelah menyentuh yang sakit.",
        "level": "danger"
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "id": "Virus Keriting Daun Kuning",
        "emoji": "🌀",
        "desc": "Daun menguning dan menggulung ke atas, tanaman menjadi kerdil dan tidak produktif.",
        "tips": "Cabut tanaman terinfeksi berat, kendalikan vektor kutu putih, gunakan benih tahan virus.",
        "level": "danger"
    },
    "Tomato_healthy": {
        "id": "Daun Sehat",
        "emoji": "✅",
        "desc": "Daun tomat dalam kondisi sehat dan tidak terdeteksi penyakit apapun.",
        "tips": "Pertahankan perawatan seperti biasa dan tetap perhatikan tanda-tanda penyakit.",
        "level": "success"
    },
}

def get_label_info(prediction_class):
    """Cari label info dengan fuzzy matching untuk handle variasi nama kelas"""
    # Coba exact match dulu
    if prediction_class in LABEL_TRANSLATE:
        return LABEL_TRANSLATE[prediction_class]
    
    # Coba normalisasi: ganti spasi/strip dan lowercase
    pred_lower = prediction_class.lower().replace(" ", "_").replace("-", "_")
    for key, val in LABEL_TRANSLATE.items():
        if key.lower().replace("-", "_") == pred_lower:
            return val
    
    # Coba partial match
    for key, val in LABEL_TRANSLATE.items():
        if pred_lower in key.lower() or key.lower() in pred_lower:
            return val
    
    return {
        "id": prediction_class,
        "emoji": "🔍",
        "desc": "Hasil deteksi dari model AI.",
        "tips": "Konsultasikan dengan ahli pertanian untuk penanganan lebih lanjut.",
        "level": "warning"
    }


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", accuracy=MODEL_ACCURACY)


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("file")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    filename = file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        # Kirim ke Roboflow API
        result = CLIENT.infer(save_path, model_id=MODEL_ID)

        # Ambil prediksi terbaik
        predictions = result.get("predictions", [])
        
        if predictions:
            # Ambil prediksi dengan confidence tertinggi
            best = max(predictions, key=lambda x: x.get("confidence", 0))
            prediction_class = best.get("class", "Unknown")
            confidence = round(best.get("confidence", 0) * 100, 1)
        else:
            # Fallback jika format berbeda (classification)
            top_class = result.get("top", "Unknown")
            confidence = round(result.get("confidence", 0) * 100, 1)
            prediction_class = top_class

        label_info = get_label_info(prediction_class)

    except Exception as e:
        return render_template(
            "result.html",
            filename=filename,
            prediction="Error Deteksi",
            emoji="❌",
            deskripsi=f"Terjadi kesalahan saat mendeteksi: {str(e)}",
            tips="Coba upload ulang gambar yang lebih jelas.",
            accuracy=MODEL_ACCURACY,
            confidence=0,
            level="danger"
        )

    return render_template(
        "result.html",
        filename=filename,
        prediction=label_info["id"],
        emoji=label_info["emoji"],
        deskripsi=label_info["desc"],
        tips=label_info["tips"],
        accuracy=MODEL_ACCURACY,
        confidence=confidence,
        level=label_info["level"]
    )


if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
