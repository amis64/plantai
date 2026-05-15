"""
Flask web application for Plant Disease Classification.
Loads the trained MobileNetV2 model and serves predictions via HTTP.

Run: python app.py
Then open: http://localhost:5000
"""

import os
import io
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'plant_disease_model.keras')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# If top prediction is below this confidence, the image is considered unrecognized.
CONFIDENCE_THRESHOLD = 0.40

# 38 classes sorted alphabetically (must match training order exactly)
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Spanish translations for all 38 classes.
# Format: "ClassName": {"plant": "...", "disease": "...", "description": "..."}
TRANSLATIONS = {
    # ── Apple ──────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "plant": "Manzana",
        "disease": "Sarna del manzano",
        "description": "Enfermedad fúngica que produce manchas oscuras y costrosas en hojas y frutos.",
    },
    "Apple___Black_rot": {
        "plant": "Manzana",
        "disease": "Pudrición negra",
        "description": "Hongo que causa pudrición en frutos y manchas foliares de color pardo-rojizo.",
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Manzana",
        "disease": "Roya del cedro-manzano",
        "description": "Enfermedad fúngica que produce manchas naranjas brillantes en las hojas.",
    },
    "Apple___healthy": {
        "plant": "Manzana",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Blueberry ──────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "plant": "Arándano",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Cherry ─────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cereza",
        "disease": "Oídio (mildiu polvoroso)",
        "description": "Hongo que cubre las hojas con un polvo blanco-grisáceo y reduce la fotosíntesis.",
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cereza",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Corn ───────────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Maíz",
        "disease": "Mancha foliar por Cercospora",
        "description": "Enfermedad fúngica que produce lesiones grises rectangulares en las hojas.",
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Maíz",
        "disease": "Roya común",
        "description": "Hongo que forma pústulas de color canela-marrón en ambas caras de la hoja.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Maíz",
        "disease": "Tizón norteño de la hoja",
        "description": "Enfermedad fúngica que produce lesiones largas y grisáceas en forma de cigarro.",
    },
    "Corn_(maize)___healthy": {
        "plant": "Maíz",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Grape ──────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "plant": "Uva",
        "disease": "Pudrición negra",
        "description": "Hongo que produce manchas foliares con borde oscuro y momificación de los frutos.",
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Uva",
        "disease": "Esca (sarampión negro)",
        "description": "Complejo de enfermedades fúngicas que causa rayas cloróticas y necrosis en hojas.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Uva",
        "disease": "Tizón foliar (mancha de Isariopsis)",
        "description": "Enfermedad fúngica que origina manchas marrones con halo amarillo en las hojas.",
    },
    "Grape___healthy": {
        "plant": "Uva",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Orange ─────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Naranja",
        "disease": "Huanglongbing (enverdecimiento de los cítricos)",
        "description": "Enfermedad bacteriana grave que provoca amarillamiento asimétrico y frutos deformes. Sin cura conocida.",
    },
    # ── Peach ──────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "plant": "Durazno",
        "disease": "Mancha bacteriana",
        "description": "Bacteria que produce pequeñas manchas angulares oscuras en hojas y lesiones en frutos.",
    },
    "Peach___healthy": {
        "plant": "Durazno",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Pepper ─────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Pimiento",
        "disease": "Mancha bacteriana",
        "description": "Bacteria que causa manchas acuosas que se tornan necróticas con halo amarillo.",
    },
    "Pepper,_bell___healthy": {
        "plant": "Pimiento",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Potato ─────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "plant": "Papa",
        "disease": "Tizón temprano",
        "description": "Hongo que produce manchas oscuras concéntricas en forma de anillos sobre las hojas.",
    },
    "Potato___Late_blight": {
        "plant": "Papa",
        "disease": "Tizón tardío",
        "description": "Enfermedad causada por oomiceto, responsable de la Gran Hambruna irlandesa. Afecta hojas, tallos y tubérculos.",
    },
    "Potato___healthy": {
        "plant": "Papa",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Raspberry ──────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "plant": "Frambuesa",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Soybean ────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "plant": "Soya",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Squash ─────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "plant": "Calabaza",
        "disease": "Oídio (mildiu polvoroso)",
        "description": "Hongo que forma un polvo blanco sobre las hojas reduciendo la capacidad fotosintética.",
    },
    # ── Strawberry ─────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "plant": "Fresa",
        "disease": "Quemadura foliar",
        "description": "Hongo que produce manchas purpúreas irregulares que se fusionan y necrosan la hoja.",
    },
    "Strawberry___healthy": {
        "plant": "Fresa",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
    # ── Tomato ─────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "plant": "Tomate",
        "disease": "Mancha bacteriana",
        "description": "Bacteria que origina pequeñas manchas acuosas oscuras en hojas, tallos y frutos.",
    },
    "Tomato___Early_blight": {
        "plant": "Tomate",
        "disease": "Tizón temprano",
        "description": "Hongo que produce manchas concéntricas oscuras rodeadas de un halo amarillo.",
    },
    "Tomato___Late_blight": {
        "plant": "Tomate",
        "disease": "Tizón tardío",
        "description": "Oomiceto que causa lesiones acuosas verde-grisáceas que destruyen rápidamente el follaje.",
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomate",
        "disease": "Moho foliar",
        "description": "Hongo que produce un moho verde-oliva en el envés de las hojas con amarillamiento superior.",
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomate",
        "disease": "Mancha foliar por Septoria",
        "description": "Hongo que causa numerosas manchas pequeñas con centro gris y borde oscuro.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomate",
        "disease": "Ácaros araña (ácaro de dos manchas)",
        "description": "Plaga de ácaros que provoca punteado amarillento y telarañas finas en las hojas.",
    },
    "Tomato___Target_Spot": {
        "plant": "Tomate",
        "disease": "Mancha en diana",
        "description": "Hongo que produce lesiones circulares con anillos concéntricos similares a una diana.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomate",
        "disease": "Virus del rizado amarillo de la hoja",
        "description": "Virus transmitido por mosca blanca que causa enrollamiento y amarillamiento severo de las hojas.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomate",
        "disease": "Virus del mosaico del tomate",
        "description": "Virus que produce un mosaico de zonas verdes claras y oscuras distorsionando el follaje.",
    },
    "Tomato___healthy": {
        "plant": "Tomate",
        "disease": "Saludable",
        "description": "La planta no presenta síntomas de enfermedad.",
    },
}


model = None


def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}. Run train.py to train the model.")
        return
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


def translate(raw_name):
    """Return Spanish plant/disease info for a class name."""
    t = TRANSLATIONS.get(raw_name)
    if t:
        return {
            'plant': t['plant'],
            'disease': t['disease'],
            'description': t['description'],
            'is_healthy': t['disease'] == 'Saludable',
        }
    # Fallback: parse raw name
    parts = raw_name.split('___', 1)
    disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Desconocida'
    return {
        'plant': parts[0].replace('_', ' ').strip(),
        'disease': disease,
        'description': '',
        'is_healthy': disease.lower() == 'healthy',
    }


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Modelo no cargado. Ejecuta train.py primero para entrenar el modelo.'
        }), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen.'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Formato no válido. Usa JPG, PNG o WebP.'}), 400

    try:
        image_bytes = file.read()
        img_array = preprocess(image_bytes)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({'error': 'No se pudo procesar la imagen. Verifica que sea una imagen válida.'}), 400

    try:
        preds = model.predict(img_array, verbose=0)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Error al hacer la predicción.'}), 500

    top5_idx = np.argsort(preds)[::-1][:5]
    top_confidence = float(preds[top5_idx[0]])

    # Below threshold: the image doesn't resemble any supported plant
    if top_confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            'success': True,
            'is_unknown': True,
            'confidence': top_confidence,
            'message': (
                'No se reconoció ninguna planta del conjunto soportado. '
                'Asegúrate de que la imagen muestre claramente la hoja de una de las '
                '14 especies admitidas.'
            ),
        })

    top5 = []
    for idx in top5_idx:
        entry = translate(CLASS_NAMES[idx])
        entry['class'] = CLASS_NAMES[idx]
        entry['confidence'] = float(preds[idx])
        top5.append(entry)

    result = {**top5[0], 'top5': top5, 'success': True, 'is_unknown': False}
    return jsonify(result)


if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000)
