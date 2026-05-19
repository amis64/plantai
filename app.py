"""
Aplicación web Flask para clasificación de enfermedades en plantas.
Carga el modelo MobileNetV2 entrenado y sirve predicciones vía HTTP.

Ejecutar: python app.py
Luego abrir: http://localhost:5000
"""

import os
import io
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

IMG_SIZE = 224
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'model', 'plant_disease_model.keras')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'class_names.json')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Si la predicción principal está por debajo de esta confianza, la imagen se considera no reconocida.
CONFIDENCE_THRESHOLD = 0.25

# Prefiere el JSON guardado por train_v2.py (soporta la taxonomía extendida).
# Usa la lista de 38 clases codificada como respaldo para compatibilidad con versiones anteriores.
def _load_class_names():
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, encoding='utf-8') as f:
                names = json.load(f)
            logger.info(f"Loaded {len(names)} classes from {CLASSES_PATH}")
            return names
        except Exception as e:
            logger.warning(f"Failed to load {CLASSES_PATH}: {e}. Using hardcoded list.")
    return _HARDCODED_CLASS_NAMES

# 38 clases ordenadas alfabéticamente (respaldo; debe coincidir exactamente con el orden de entrenamiento v1)
_HARDCODED_CLASS_NAMES = [
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

CLASS_NAMES = _load_class_names()

# Traducciones al español de todas las clases soportadas.
# Formato: "NombreClase": {"plant": "...", "disease": "...", "description": "..."}
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
    # ── Apple (extended) ───────────────────────────────────────────────────────
    "Apple___Mosaic_virus": {
        "plant": "Manzana",
        "disease": "Virus del mosaico",
        "description": "Virus que produce un mosaico de zonas claras y oscuras en las hojas.",
    },
    # ── Banana ────────────────────────────────────────────────────────────────
    "Banana___Anthracnose": {
        "plant": "Banana",
        "disease": "Antracnosis",
        "description": "Hongo que causa manchas oscuras en hojas y frutos, pudriéndolos en poscosecha.",
    },
    "Banana___Black_sigatoka": {
        "plant": "Banana",
        "disease": "Sigatoka negra",
        "description": "Enfermedad fúngica grave que destruye el área foliar y reduce la producción hasta un 50%.",
    },
    "Banana___Bunchy_top": {
        "plant": "Banana",
        "disease": "Cogollo racimoso",
        "description": "Virus transmitido por pulgones que produce hojas pequeñas y erizadas en el cogollo.",
    },
    "Banana___Cigar_end_rot": {
        "plant": "Banana",
        "disease": "Pudrición del extremo",
        "description": "Hongo que provoca una pudrición seca de color negro en el extremo del fruto.",
    },
    "Banana___Cordana_leaf_spot": {
        "plant": "Banana",
        "disease": "Mancha foliar Cordana",
        "description": "Enfermedad fúngica que forma manchas ovaladas con halo amarillo en las hojas.",
    },
    "Banana___Panama_disease": {
        "plant": "Banana",
        "disease": "Mal de Panamá",
        "description": "Enfermedad vascular causada por Fusarium que marchita y mata la planta sin cura conocida.",
    },
    # ── Basil ─────────────────────────────────────────────────────────────────
    "Basil___Downy_mildew": {
        "plant": "Albahaca",
        "disease": "Mildiu velloso",
        "description": "Oomiceto que produce un polvo grisáceo-morado en el envés de las hojas.",
    },
    # ── Bean ──────────────────────────────────────────────────────────────────
    "Bean___Halo_blight": {
        "plant": "Frijol",
        "disease": "Tizón aureolado",
        "description": "Bacteria que forma manchas con un halo amarillo claro característico en las hojas.",
    },
    "Bean___Mosaic_virus": {
        "plant": "Frijol",
        "disease": "Virus del mosaico",
        "description": "Virus que produce mosaico de colores y deformación foliar, reduciendo el rendimiento.",
    },
    "Bean___Rust": {
        "plant": "Frijol",
        "disease": "Roya",
        "description": "Hongo que forma pústulas de color óxido en ambas caras de las hojas.",
    },
    # ── Blueberry (extended) ───────────────────────────────────────────────────
    "Blueberry___Anthracnose": {
        "plant": "Arándano",
        "disease": "Antracnosis",
        "description": "Hongo que provoca manchas deprimidas y pudrición en frutos maduros.",
    },
    "Blueberry___Botrytis_blight": {
        "plant": "Arándano",
        "disease": "Moho gris (Botrytis)",
        "description": "Hongo que cubre flores y frutos con un moho gris harinoso bajo condiciones húmedas.",
    },
    "Blueberry___Mummy_berry": {
        "plant": "Arándano",
        "disease": "Momificación del fruto",
        "description": "Enfermedad fúngica que marchita brotes y momifica frutos dejándolos de color salmón.",
    },
    "Blueberry___Rust": {
        "plant": "Arándano",
        "disease": "Roya",
        "description": "Hongo que produce pústulas anaranjadas en el envés de las hojas.",
    },
    "Blueberry___Scorch": {
        "plant": "Arándano",
        "disease": "Quemadura foliar",
        "description": "Enfermedad viral que causa necrosis de flores, hojas y ramas simulando quemaduras.",
    },
    # ── Broccoli ──────────────────────────────────────────────────────────────
    "Broccoli___Alternaria_leaf_spot": {
        "plant": "Brócoli",
        "disease": "Mancha foliar Alternaria",
        "description": "Hongo que produce manchas circulares oscuras con anillos concéntricos en las hojas.",
    },
    "Broccoli___Downy_mildew": {
        "plant": "Brócoli",
        "disease": "Mildiu velloso",
        "description": "Oomiceto que provoca manchas amarillas en el haz y pelusa gris en el envés.",
    },
    # ── Cabbage ───────────────────────────────────────────────────────────────
    "Cabbage___Alternaria_leaf_spot": {
        "plant": "Col",
        "disease": "Mancha foliar Alternaria",
        "description": "Hongo que forma manchas oscuras concéntricas en las hojas externas de la col.",
    },
    "Cabbage___Black_rot": {
        "plant": "Col",
        "disease": "Pudrición negra",
        "description": "Bacteria que causa amarillamiento en forma de V y ennegrecimiento de nervaduras.",
    },
    "Cabbage___Downy_mildew": {
        "plant": "Col",
        "disease": "Mildiu velloso",
        "description": "Oomiceto que produce manchas amarillas y crecimiento fúngico en el envés foliar.",
    },
    # ── Carrot ────────────────────────────────────────────────────────────────
    "Carrot___Alternaria_leaf_blight": {
        "plant": "Zanahoria",
        "disease": "Tizón foliar Alternaria",
        "description": "Hongo que produce manchas oscuras irregulares con halo amarillo en el follaje.",
    },
    "Carrot___Cavity_spot": {
        "plant": "Zanahoria",
        "disease": "Mancha cavitaria",
        "description": "Enfermedad que forma cavidades elípticas en la superficie de la raíz.",
    },
    # ── Cauliflower ───────────────────────────────────────────────────────────
    "Cauliflower___Alternaria_leaf_spot": {
        "plant": "Coliflor",
        "disease": "Mancha foliar Alternaria",
        "description": "Hongo que genera manchas concéntricas oscuras en hojas y pueden llegar a las pellas.",
    },
    "Cauliflower___Bacterial_soft_rot": {
        "plant": "Coliflor",
        "disease": "Pudrición blanda bacteriana",
        "description": "Bacteria que provoca una pudrición acuosa maloliente en la pella y el tallo.",
    },
    # ── Celery ────────────────────────────────────────────────────────────────
    "Celery___Early_blight": {
        "plant": "Apio",
        "disease": "Tizón temprano",
        "description": "Hongo que produce manchas amarillas que se tornan marrones con esporulación central.",
    },
    # ── Cherry (extended) ─────────────────────────────────────────────────────
    "Cherry_(including_sour)___Leaf_spot": {
        "plant": "Cereza",
        "disease": "Mancha foliar",
        "description": "Hongo que genera manchas purpúreas que evolucionan a necrosis y caída prematura de hojas.",
    },
    # ── Coffee ────────────────────────────────────────────────────────────────
    "Coffee___Berry_blotch": {
        "plant": "Café",
        "disease": "Mancha del fruto",
        "description": "Enfermedad que produce manchas oscuras en los frutos del café afectando su calidad.",
    },
    "Coffee___Brown_eye_spot": {
        "plant": "Café",
        "disease": "Mancha ojo pardo",
        "description": "Hongo que forma manchas circulares con centro claro y borde oscuro en las hojas.",
    },
    "Coffee___Leaf_rust": {
        "plant": "Café",
        "disease": "Roya del cafeto",
        "description": "Hongo devastador que produce polvillo anaranjado en el envés reduciendo drásticamente la cosecha.",
    },
    # ── Corn (extended) ───────────────────────────────────────────────────────
    "Corn_(maize)___Smut": {
        "plant": "Maíz",
        "disease": "Carbón del maíz",
        "description": "Hongo que forma agallas blancas que se tornan negras y polvorientas al madurar.",
    },
    # ── Cucumber ──────────────────────────────────────────────────────────────
    "Cucumber___Angular_leaf_spot": {
        "plant": "Pepino",
        "disease": "Mancha angular de la hoja",
        "description": "Bacteria que produce lesiones angulares acuosas delimitadas por nervaduras que se necrosan.",
    },
    "Cucumber___Bacterial_wilt": {
        "plant": "Pepino",
        "disease": "Marchitez bacteriana",
        "description": "Bacteria transmitida por diabróticas que obstruye el xilema causando marchitez y muerte rápida.",
    },
    "Cucumber___Powdery_mildew": {
        "plant": "Pepino",
        "disease": "Oídio",
        "description": "Hongo que cubre las hojas con un polvo blanco reduciendo la fotosíntesis y el rendimiento.",
    },
    # ── Eggplant ──────────────────────────────────────────────────────────────
    "Eggplant___Cercospora_leaf_spot": {
        "plant": "Berenjena",
        "disease": "Mancha foliar Cercospora",
        "description": "Hongo que produce manchas circulares con centro grisáceo y borde oscuro bien definido.",
    },
    "Eggplant___Phomopsis_fruit_rot": {
        "plant": "Berenjena",
        "disease": "Pudrición del fruto por Phomopsis",
        "description": "Hongo que causa pudrición blanda y hundimiento del fruto con lesiones pardas.",
    },
    "Eggplant___Phytophthora_blight": {
        "plant": "Berenjena",
        "disease": "Tizón por Phytophthora",
        "description": "Oomiceto que provoca lesiones acuosas en tallos, hojas y frutos con colapso rápido.",
    },
    # ── Garlic ────────────────────────────────────────────────────────────────
    "Garlic___Leaf_blight": {
        "plant": "Ajo",
        "disease": "Tizón foliar",
        "description": "Hongo que produce manchas blanquecinas que evolucionan a necrosis de las puntas foliares.",
    },
    "Garlic___Rust": {
        "plant": "Ajo",
        "disease": "Roya",
        "description": "Hongo que forma pústulas anaranjadas en ambas caras de las hojas del ajo.",
    },
    # ── Grape (extended) ──────────────────────────────────────────────────────
    "Grape___Downy_mildew": {
        "plant": "Uva",
        "disease": "Mildiu velloso",
        "description": "Oomiceto que produce manchas aceitosas en el haz y pelusa blanca en el envés.",
    },
    "Grape___Leaf_spot": {
        "plant": "Uva",
        "disease": "Mancha foliar",
        "description": "Enfermedad fúngica que produce manchas oscuras en las hojas de la vid.",
    },
    "Grape___Leafroll_disease": {
        "plant": "Uva",
        "disease": "Enrollamiento de la hoja",
        "description": "Virus que causa enrollamiento hacia abajo de las hojas y reducción de azúcares en el fruto.",
    },
    # ── Lettuce ───────────────────────────────────────────────────────────────
    "Lettuce___Downy_mildew": {
        "plant": "Lechuga",
        "disease": "Mildiu velloso",
        "description": "Oomiceto que produce manchas amarillas angulares y pelusa gris en el envés foliar.",
    },
    # ── Maple ─────────────────────────────────────────────────────────────────
    "Maple___Tar_spot": {
        "plant": "Arce",
        "disease": "Mancha de alquitrán",
        "description": "Hongo que forma manchas negras brillantes de aspecto similar al alquitrán en las hojas.",
    },
    # ── Orange (extended) ─────────────────────────────────────────────────────
    "Orange___Citrus_canker": {
        "plant": "Naranja",
        "disease": "Cancro de los cítricos",
        "description": "Bacteria que produce lesiones corchosas con halo amarillo en hojas, tallos y frutos.",
    },
    # ── Peach (extended) ──────────────────────────────────────────────────────
    "Peach___Brown_rot": {
        "plant": "Durazno",
        "disease": "Pudrición parda",
        "description": "Hongo que causa pudrición rápida del fruto con cobertura de esporas grises en condiciones húmedas.",
    },
    "Peach___Leaf_curl": {
        "plant": "Durazno",
        "disease": "Abolladura del durazno",
        "description": "Hongo que provoca ondulación, engrosamiento y enrojecimiento de las hojas en primavera.",
    },
    "Peach___Scab": {
        "plant": "Durazno",
        "disease": "Sarna del durazno",
        "description": "Hongo que produce manchas verde-oliva en frutos jóvenes que se tornan corchosas.",
    },
    # ── Pepper (extended) ─────────────────────────────────────────────────────
    "Pepper,_bell___Blossom_end_rot": {
        "plant": "Pimiento",
        "disease": "Pudrición apical",
        "description": "Desorden fisiológico por déficit de calcio que causa necrosis oscura en el extremo del fruto.",
    },
    "Pepper,_bell___Frogeye_leaf_spot": {
        "plant": "Pimiento",
        "disease": "Mancha ojo de rana",
        "description": "Hongo que forma manchas circulares con centro pálido y borde oscuro bien definido.",
    },
    "Pepper,_bell___Powdery_mildew": {
        "plant": "Pimiento",
        "disease": "Oídio",
        "description": "Hongo que cubre las hojas con un polvillo blanco reduciendo la capacidad fotosintética.",
    },
    # ── Plum ──────────────────────────────────────────────────────────────────
    "Plum___Brown_rot": {
        "plant": "Ciruela",
        "disease": "Pudrición parda",
        "description": "Hongo que provoca pudrición rápida de flores y frutos con esporulación grisácea.",
    },
    "Plum___Pocket_disease": {
        "plant": "Ciruela",
        "disease": "Bolsillos de la ciruela",
        "description": "Hongo que deforma el fruto creando sacos alargados sin hueso llenos de esporas.",
    },
    # ── Raspberry (extended) ──────────────────────────────────────────────────
    "Raspberry___Gray_mold": {
        "plant": "Frambuesa",
        "disease": "Moho gris",
        "description": "Hongo Botrytis que pudre flores y frutos cubiertos de moho gris en condiciones húmedas.",
    },
    # ── Rice ──────────────────────────────────────────────────────────────────
    "Rice___Leaf_blast": {
        "plant": "Arroz",
        "disease": "Piricularia del arroz",
        "description": "Hongo devastador que produce lesiones en forma de diamante en hojas, nudos y panículas.",
    },
    "Rice___Sheath_blight": {
        "plant": "Arroz",
        "disease": "Tizón de la vaina",
        "description": "Hongo que forma lesiones irregulares verdosas en la vaina y puede colapsar la planta.",
    },
    # ── Soybean (extended) ────────────────────────────────────────────────────
    "Soybean___Bacterial_blight": {
        "plant": "Soya",
        "disease": "Tizón bacteriano",
        "description": "Bacteria que produce manchas angulares acuosas que se vuelven pardas con halo amarillo.",
    },
    "Soybean___Brown_spot": {
        "plant": "Soya",
        "disease": "Mancha parda",
        "description": "Hongo que genera manchas irregulares de color marrón en hojas inferiores primero.",
    },
    "Soybean___Downy_mildew": {
        "plant": "Soya",
        "disease": "Mildiu velloso",
        "description": "Oomiceto que produce manchas grises en el haz y pelusa blanca en el envés de las hojas.",
    },
    "Soybean___Frogeye_leaf_spot": {
        "plant": "Soya",
        "disease": "Mancha ojo de rana",
        "description": "Hongo que forma manchas con centro gris claro y borde oscuro rodeadas de halo amarillo.",
    },
    "Soybean___Mosaic": {
        "plant": "Soya",
        "disease": "Mosaico de la soya",
        "description": "Virus que provoca mosaico foliar, arrugamiento y reducción en el tamaño de granos.",
    },
    "Soybean___Rust": {
        "plant": "Soya",
        "disease": "Roya de la soya",
        "description": "Hongo que produce pústulas pequeñas en el envés con lesiones amarillas en el haz.",
    },
    # ── Strawberry (extended) ─────────────────────────────────────────────────
    "Strawberry___Anthracnose": {
        "plant": "Fresa",
        "disease": "Antracnosis",
        "description": "Hongo que causa manchas oscuras hundidas en frutos y lesiones acuosas en tallos.",
    },
    # ── Tobacco ───────────────────────────────────────────────────────────────
    "Tobacco___Blue_mold": {
        "plant": "Tabaco",
        "disease": "Moho azul",
        "description": "Oomiceto que produce manchas amarillas y pelusa azulada en el envés de las hojas.",
    },
    "Tobacco___Mosaic_virus": {
        "plant": "Tabaco",
        "disease": "Virus del mosaico del tabaco",
        "description": "Virus altamente resistente que produce mosaico foliar y necrosis sistémica en la planta.",
    },
    # ── Wheat ─────────────────────────────────────────────────────────────────
    "Wheat___Bacterial_leaf_streak": {
        "plant": "Trigo",
        "disease": "Rayado bacteriano",
        "description": "Bacteria que produce rayas acuosas translúcidas que se vuelven pardas al secarse.",
    },
    "Wheat___Head_scab": {
        "plant": "Trigo",
        "disease": "Fusariosis de la espiga",
        "description": "Hongo que blanquea espigas parcialmente y produce micotoxinas dañinas en el grano.",
    },
    "Wheat___Leaf_rust": {
        "plant": "Trigo",
        "disease": "Roya de la hoja",
        "description": "Hongo que forma pústulas anaranjadas circulares en el haz de las hojas.",
    },
    "Wheat___Loose_smut": {
        "plant": "Trigo",
        "disease": "Carbón suelto",
        "description": "Hongo que reemplaza completamente el grano por una masa de esporas negras.",
    },
    "Wheat___Powdery_mildew": {
        "plant": "Trigo",
        "disease": "Oídio del trigo",
        "description": "Hongo que cubre hojas y tallos con un polvo blanco reduciendo la fotosíntesis.",
    },
    "Wheat___Septoria_blotch": {
        "plant": "Trigo",
        "disease": "Septoriosis del trigo",
        "description": "Hongo que produce manchas necróticas con picnidios negros que reducen el área foliar.",
    },
    "Wheat___Stem_rust": {
        "plant": "Trigo",
        "disease": "Roya del tallo",
        "description": "Hongo que forma pústulas rojo-oxidadas en tallos y vainas pudiendo tumbar el cultivo.",
    },
    "Wheat___Stripe_rust": {
        "plant": "Trigo",
        "disease": "Roya amarilla",
        "description": "Hongo que produce rayas de pústulas amarillas a lo largo de las hojas en clima fresco.",
    },
    # ── Zucchini ──────────────────────────────────────────────────────────────
    "Zucchini___Bacterial_wilt": {
        "plant": "Calabacín",
        "disease": "Marchitez bacteriana",
        "description": "Bacteria transmitida por escarabajos de pepino que obstruye el xilema causando marchitez.",
    },
    "Zucchini___Downy_mildew": {
        "plant": "Calabacín",
        "disease": "Mildiu velloso",
        "description": "Oomiceto que produce manchas angulares amarillas y pelusa gris-violeta en el envés.",
    },
    "Zucchini___Powdery_mildew": {
        "plant": "Calabacín",
        "disease": "Oídio",
        "description": "Hongo que cubre las hojas con polvo blanco reduciendo el vigor y la producción.",
    },
    "Zucchini___Yellow_mosaic_virus": {
        "plant": "Calabacín",
        "disease": "Virus del mosaico amarillo",
        "description": "Virus transmitido por pulgones que deforma hojas y frutos con mosaico amarillo-verde.",
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
    """Devuelve la información de planta/enfermedad en español para un nombre de clase."""
    t = TRANSLATIONS.get(raw_name)
    if t:
        return {
            'plant': t['plant'],
            'disease': t['disease'],
            'description': t['description'],
            'is_healthy': t['disease'] == 'Saludable',
        }
    # Respaldo: parsear el nombre crudo
    parts = raw_name.split('___', 1)
    disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Desconocida'
    return {
        'plant': parts[0].replace('_', ' ').strip(),
        'disease': disease,
        'description': '',
        'is_healthy': disease.lower() == 'healthy',
    }


def _build_plant_class_map():
    m = {}
    for idx, cls in enumerate(CLASS_NAMES):
        plant = translate(cls)['plant']
        m.setdefault(plant, []).append(idx)
    return m

PLANT_CLASS_MAP: dict = _build_plant_class_map()


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


@app.route('/classes')
def get_classes():
    plants = [
        {'name': plant, 'count': len(indices)}
        for plant, indices in sorted(PLANT_CLASS_MAP.items())
    ]
    return jsonify({'plants': plants, 'total': len(CLASS_NAMES)})


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

    # Filtro de planta opcional (modo guiado)
    plant_filter = request.form.get('plant', '').strip() or None

    try:
        preds = model.predict(img_array, verbose=0)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Error al hacer la predicción.'}), 500

    # ── Modo guiado: enmascarar probabilidades a las clases de la planta seleccionada ──
    if plant_filter and plant_filter in PLANT_CLASS_MAP:
        mask_indices = PLANT_CLASS_MAP[plant_filter]
        masked = np.zeros_like(preds)
        for i in mask_indices:
            masked[i] = preds[i]
        total = float(masked.sum())
        preds = masked / total if total > 0 else masked
        effective_threshold = 0.05   # muy bajo — la planta ya es conocida
        top5_pool = sorted(mask_indices, key=lambda i: preds[i], reverse=True)
    else:
        plant_filter = None
        effective_threshold = CONFIDENCE_THRESHOLD
        top5_pool = list(np.argsort(preds)[::-1])

    top5_idx = top5_pool[:5]
    top_confidence = float(preds[top5_idx[0]])

    if top_confidence < effective_threshold:
        if plant_filter:
            msg = (
                f'La imagen no parece mostrar una hoja de {plant_filter} con claridad. '
                'Intenta con mejor iluminación o un ángulo más cercano a la hoja.'
            )
        else:
            msg = (
                'No se reconoció ninguna planta del conjunto soportado. '
                'Asegúrate de que la imagen muestre claramente la hoja de la planta '
                'con buena iluminación y sin objetos que la obstruyan.'
            )
        return jsonify({
            'success': True,
            'is_unknown': True,
            'confidence': top_confidence,
            'message': msg,
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
