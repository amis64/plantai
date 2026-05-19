"""
build_dataset.py
Parsea los 3 datasets con sus formatos nativos de etiquetas, aplica mapeos
curados de clases y escribe data/manifest.csv con columnas:
    path, label, split, source

Ejecutar una vez antes del entrenamiento:
    python build_dataset.py
    python build_dataset.py --npd <ruta> --plantseg <ruta> --plantdec <ruta>
"""

import os
import csv
import json
import yaml
import argparse
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'data', 'manifest.csv')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MIN_PER_CLASS = 25   # eliminar clases con menos imágenes que este umbral


# ═══════════════════════════════════════════════════════════════════════════════
#  MAPEOS CURADOS DE ETIQUETAS
# ═══════════════════════════════════════════════════════════════════════════════

# PlantSeg: (planta, cadena_enfermedad) → etiqueta unificada
# Clave: coincidencia exacta en minúsculas de las columnas Plant + Disease del CSV
PLANTSEG_MAP = {
    # ── Manzana ────────────────────────────────────────────────────────────────
    ('apple', 'apple black rot'):          'Apple___Black_rot',
    ('apple', 'apple mosaic virus'):       'Apple___Mosaic_virus',
    ('apple', 'apple rust'):               'Apple___Cedar_apple_rust',
    ('apple', 'apple scab'):               'Apple___Apple_scab',
    # ── Banana ────────────────────────────────────────────────────────────────
    ('banana', 'banana anthracnose'):      'Banana___Anthracnose',
    ('banana', 'banana black leaf streak'):'Banana___Black_sigatoka',
    ('banana', 'banana bunchy top'):       'Banana___Bunchy_top',
    ('banana', 'banana cigar end rot'):    'Banana___Cigar_end_rot',
    ('banana', 'banana cordana leaf spot'):'Banana___Cordana_leaf_spot',
    ('banana', 'banana panama disease'):   'Banana___Panama_disease',
    # ── Albahaca ──────────────────────────────────────────────────────────────
    ('basil', 'basil downy mildew'):       'Basil___Downy_mildew',
    # ── Frijol ────────────────────────────────────────────────────────────────
    ('bean', 'bean halo blight'):          'Bean___Halo_blight',
    ('bean', 'bean mosaic virus'):         'Bean___Mosaic_virus',
    ('bean', 'bean rust'):                 'Bean___Rust',
    # ── Pimiento ──────────────────────────────────────────────────────────────
    ('bell pepper', 'bell pepper bacterial spot'):     'Pepper,_bell___Bacterial_spot',
    ('bell pepper', 'bell pepper blossom end rot'):    'Pepper,_bell___Blossom_end_rot',
    ('bell pepper', 'bell pepper frogeye leaf spot'):  'Pepper,_bell___Frogeye_leaf_spot',
    ('bell pepper', 'bell pepper powdery mildew'):     'Pepper,_bell___Powdery_mildew',
    # ── Arándano ──────────────────────────────────────────────────────────────
    ('blueberry', 'blueberry anthracnose'):  'Blueberry___Anthracnose',
    ('blueberry', 'blueberry botrytis blight'): 'Blueberry___Botrytis_blight',
    ('blueberry', 'blueberry mummy berry'): 'Blueberry___Mummy_berry',
    ('blueberry', 'blueberry rust'):         'Blueberry___Rust',
    ('blueberry', 'blueberry scorch'):       'Blueberry___Scorch',
    # ── Brócoli ───────────────────────────────────────────────────────────────
    ('broccoli', 'broccoli alternaria leaf spot'): 'Broccoli___Alternaria_leaf_spot',
    ('broccoli', 'broccoli downy mildew'):    'Broccoli___Downy_mildew',
    # ── Col ───────────────────────────────────────────────────────────────────
    ('cabbage', 'cabbage alternaria leaf spot'): 'Cabbage___Alternaria_leaf_spot',
    ('cabbage', 'cabbage black rot'):         'Cabbage___Black_rot',
    ('cabbage', 'cabbage downy mildew'):      'Cabbage___Downy_mildew',
    # ── Zanahoria ─────────────────────────────────────────────────────────────
    ('carrot', 'carrot alternaria leaf blight'): 'Carrot___Alternaria_leaf_blight',
    ('carrot', 'carrot cavity spot'):         'Carrot___Cavity_spot',
    # ── Coliflor ──────────────────────────────────────────────────────────────
    ('cauliflower', 'cauliflower alternaria leaf spot'): 'Cauliflower___Alternaria_leaf_spot',
    ('cauliflower', 'cauliflower bacterial soft rot'):   'Cauliflower___Bacterial_soft_rot',
    # ── Apio ──────────────────────────────────────────────────────────────────
    ('celery', 'celery early blight'):        'Celery___Early_blight',
    # ── Cereza ────────────────────────────────────────────────────────────────
    ('cherry', 'cherry leaf spot'):           'Cherry_(including_sour)___Leaf_spot',
    ('cherry', 'cherry powdery mildew'):      'Cherry_(including_sour)___Powdery_mildew',
    # ── Naranja / Cítrico ─────────────────────────────────────────────────────
    ('citrus', 'citrus canker'):              'Orange___Citrus_canker',
    ('citrus', 'citrus greening disease'):    'Orange___Haunglongbing_(Citrus_greening)',
    # ── Café ──────────────────────────────────────────────────────────────────
    ('coffee', 'coffee berry blotch'):        'Coffee___Berry_blotch',
    ('coffee', 'coffee brown eye spot'):      'Coffee___Brown_eye_spot',
    ('coffee', 'coffee leaf rust'):           'Coffee___Leaf_rust',
    # ── Maíz ──────────────────────────────────────────────────────────────────
    ('corn', 'corn gray leaf spot'):          'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    ('corn', 'corn northern leaf blight'):    'Corn_(maize)___Northern_Leaf_Blight',
    ('corn', 'corn rust'):                    'Corn_(maize)___Common_rust_',
    ('corn', 'corn smut'):                    'Corn_(maize)___Smut',
    # ── Pepino ────────────────────────────────────────────────────────────────
    ('cucumber', 'cucumber angular leaf spot'): 'Cucumber___Angular_leaf_spot',
    ('cucumber', 'cucumber bacterial wilt'):    'Cucumber___Bacterial_wilt',
    ('cucumber', 'cucumber powdery mildew'):    'Cucumber___Powdery_mildew',
    # ── Berenjena ─────────────────────────────────────────────────────────────
    ('eggplant', 'eggplant cercospora leaf spot'): 'Eggplant___Cercospora_leaf_spot',
    ('eggplant', 'eggplant phomopsis fruit rot'):  'Eggplant___Phomopsis_fruit_rot',
    ('eggplant', 'eggplant phytophthora blight'):  'Eggplant___Phytophthora_blight',
    # ── Ajo ───────────────────────────────────────────────────────────────────
    ('garlic', 'garlic leaf blight'):         'Garlic___Leaf_blight',
    ('garlic', 'garlic rust'):                'Garlic___Rust',
    # ── Uva ───────────────────────────────────────────────────────────────────
    ('grape', 'grape black rot'):             'Grape___Black_rot',
    ('grape', 'grape downy mildew'):          'Grape___Downy_mildew',
    ('grape', 'grape leaf spot'):             'Grape___Leaf_spot',
    ('grapevine', 'grapevine leafroll disease'): 'Grape___Leafroll_disease',
    # ── Lechuga ───────────────────────────────────────────────────────────────
    ('lettuce', 'lettuce downy mildew'):      'Lettuce___Downy_mildew',
    # ── Arce ──────────────────────────────────────────────────────────────────
    ('maple', 'maple tar spot'):              'Maple___Tar_spot',
    # ── Durazno ───────────────────────────────────────────────────────────────
    ('peach', 'peach brown rot'):             'Peach___Brown_rot',
    ('peach', 'peach leaf curl'):             'Peach___Leaf_curl',
    ('peach', 'peach scab'):                  'Peach___Scab',
    # ── Ciruela ───────────────────────────────────────────────────────────────
    ('plum', 'plum brown rot'):               'Plum___Brown_rot',
    ('plum', 'plum pocket disease'):          'Plum___Pocket_disease',
    # ── Papa ──────────────────────────────────────────────────────────────────
    ('potato', 'potato early blight'):        'Potato___Early_blight',
    ('potato', 'potato late blight'):         'Potato___Late_blight',
    # ── Frambuesa ─────────────────────────────────────────────────────────────
    ('raspberry', 'raspberry gray mold'):     'Raspberry___Gray_mold',
    # ── Arroz ─────────────────────────────────────────────────────────────────
    ('rice', 'rice blast'):                   'Rice___Leaf_blast',
    ('rice', 'rice sheath blight'):           'Rice___Sheath_blight',
    # ── Soya ──────────────────────────────────────────────────────────────────
    ('soybean', 'bean rust'):                 'Soybean___Rust',
    ('soybean', 'soybean bacterial blight'):  'Soybean___Bacterial_blight',
    ('soybean', 'soybean brown spot'):        'Soybean___Brown_spot',
    ('soybean', 'soybean downy mildew'):      'Soybean___Downy_mildew',
    ('soybean', 'soybean frog eye leaf spot'):'Soybean___Frogeye_leaf_spot',
    ('soybean', 'soybean mosaic'):            'Soybean___Mosaic',
    # ── Calabaza ──────────────────────────────────────────────────────────────
    ('squash', 'squash powdery mildew'):      'Squash___Powdery_mildew',
    # ── Fresa ─────────────────────────────────────────────────────────────────
    ('strawberry', 'strawberry anthracnose'): 'Strawberry___Anthracnose',
    ('strawberry', 'strawberry leaf scorch'): 'Strawberry___Leaf_scorch',
    # ── Tabaco ────────────────────────────────────────────────────────────────
    ('tobacco', 'tobacco mosaic virus'):      'Tobacco___Mosaic_virus',
    ('tobacco', 'tobacco blue mold'):         'Tobacco___Blue_mold',
    # ── Tomate ────────────────────────────────────────────────────────────────
    ('tomato', 'tomato bacterial leaf spot'): 'Tomato___Bacterial_spot',
    ('tomato', 'tomato early blight'):        'Tomato___Early_blight',
    ('tomato', 'tomato late blight'):         'Tomato___Late_blight',
    ('tomato', 'tomato leaf mold'):           'Tomato___Leaf_Mold',
    ('tomato', 'tomato mosaic virus'):        'Tomato___Tomato_mosaic_virus',
    ('tomato', 'tomato septoria leaf spot'):  'Tomato___Septoria_leaf_spot',
    ('tomato', 'tomato yellow leaf curl virus'): 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    # ── Trigo ─────────────────────────────────────────────────────────────────
    ('wheat', 'wheat head scab'):             'Wheat___Head_scab',
    ('wheat', 'wheat leaf rust'):             'Wheat___Leaf_rust',
    ('wheat', 'wheat loose smut'):            'Wheat___Loose_smut',
    ('wheat', 'wheat powdery mildew'):        'Wheat___Powdery_mildew',
    ('wheat', 'wheat septoria blotch'):       'Wheat___Septoria_blotch',
    ('wheat', 'wheat stem rust'):             'Wheat___Stem_rust',
    ('wheat', 'wheat stripe rust'):           'Wheat___Stripe_rust',
    ('wheat', 'wheat bacterial leaf streak (black chaff)'): 'Wheat___Bacterial_leaf_streak',
    # ── Calabacín ─────────────────────────────────────────────────────────────
    ('zucchini', 'zucchini bacterial wilt'):  'Zucchini___Bacterial_wilt',
    ('zucchini', 'zucchini downy mildew'):    'Zucchini___Downy_mildew',
    ('zucchini', 'zucchini powdery mildew'):  'Zucchini___Powdery_mildew',
    ('zucchini', 'zucchini yellow mosaic virus'): 'Zucchini___Yellow_mosaic_virus',
}

# PlantDEC: índice de clase YOLO → etiqueta unificada
# Los nombres de clase vienen de data.yaml; mapeados aquí explícitamente para evitar errores de coincidencia aproximada.
PLANTDEC_CLASS_MAP = [
    'Apple___Apple_scab',                                      # 0  Sarna de hoja de manzana
    'Apple___healthy',                                         # 1  Hoja de manzana sana
    'Apple___Cedar_apple_rust',                                # 2  Roya de hoja de manzana
    'Pepper,_bell___Bacterial_spot',                           # 3  Mancha bacteriana en pimiento
    'Pepper,_bell___healthy',                                  # 4  Hoja de pimiento sana
    'Blueberry___healthy',                                     # 5  Hoja de arándano sana
    'Cherry_(including_sour)___healthy',                       # 6  Hoja de cereza sana
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',      # 7  Mancha gris de Cercospora en maíz
    'Corn_(maize)___Northern_Leaf_Blight',                     # 8  Tizón norteño de la hoja de maíz
    'Corn_(maize)___Common_rust_',                             # 9  Roya común del maíz
    'Peach___healthy',                                         # 10 Hoja de durazno sana
    'Potato___Early_blight',                                   # 11 Tizón temprano en papa
    'Potato___Late_blight',                                    # 12 Tizón tardío en papa
    'Potato___healthy',                                        # 13 Hoja de papa sana
    'Raspberry___healthy',                                     # 14 Hoja de frambuesa sana
    'Soybean___healthy',                                       # 15 Hoja de soya sana (variante 1)
    'Soybean___healthy',                                       # 16 Hoja de soya sana (variante 2)
    'Squash___Powdery_mildew',                                 # 17 Oídio en calabaza
    'Strawberry___healthy',                                    # 18 Hoja de fresa sana
    'Tomato___Early_blight',                                   # 19 Tizón temprano en tomate
    'Tomato___Septoria_leaf_spot',                             # 20 Mancha de Septoria en tomate
    'Tomato___Bacterial_spot',                                 # 21 Mancha bacteriana en tomate
    'Tomato___Late_blight',                                    # 22 Tizón tardío en tomate
    'Tomato___Tomato_mosaic_virus',                            # 23 Virus del mosaico del tomate
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',                  # 24 Virus del rizado amarillo del tomate
    'Tomato___healthy',                                        # 25 Hoja de tomate sana
    'Tomato___Leaf_Mold',                                      # 26 Moho foliar del tomate
    'Tomato___Spider_mites Two-spotted_spider_mite',           # 27 Ácaros araña de dos manchas en tomate
    'Grape___Black_rot',                                       # 28 Pudrición negra de la uva
    'Grape___healthy',                                         # 29 Hoja de uva sana
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def parse_npd(npd_root: str) -> list:
    """
    New Plant Diseases Dataset — estructura estándar de una carpeta por clase.
    Los subdirectorios train/ y valid/ contienen las carpetas de clase.
    """
    entries = []
    for split_name, split_dir in [('train', 'train'), ('val', 'valid')]:
        split_path = os.path.join(npd_root, split_dir)
        if not os.path.isdir(split_path):
            continue
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                if _is_image(fpath):
                    entries.append({
                        'path':   fpath,
                        'label':  cls,
                        'split':  split_name,
                        'source': 'npd',
                    })
    log.info(f"NPD: {len(entries):,} imágenes desde {npd_root}")
    return entries


def parse_plantseg(plantseg_root: str) -> list:
    """
    PlantSeg — dataset de segmentación con etiquetas en Metadatav2.csv.
    Las imágenes están en plantsegv2/images/{train|val|test}/
    Las etiquetas vienen del CSV (columnas Plant + Disease).
    """
    # Localizar CSV y raíz de imágenes
    csv_path = os.path.join(plantseg_root, 'plantsegv2', 'Metadatav2.csv')
    # Manejar el prefijo anidado '1/' que kagglehub a veces agrega
    if not os.path.exists(csv_path):
        csv_path = os.path.join(plantseg_root, '1', 'plantsegv2', 'Metadatav2.csv')
    img_root = os.path.join(os.path.dirname(csv_path), 'images')

    if not os.path.exists(csv_path):
        log.warning(f"CSV de PlantSeg no encontrado, omitiendo. Buscado en: {csv_path}")
        return []

    split_map = {'training': 'train', 'validation': 'val', 'test': 'test'}
    unmapped, entries = [], []

    with open(csv_path, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plant   = row['Plant'].strip().lower()
            disease = row['Disease'].strip().lower()
            fname   = row['Name'].strip()
            split   = split_map.get(row['Split'].strip().lower(), 'train')

            # Mapear (planta, enfermedad) → etiqueta unificada
            label = PLANTSEG_MAP.get((plant, disease))
            if label is None:
                unmapped.append((plant, disease))
                continue

            # Localizar el archivo de imagen en el directorio del split
            fpath = os.path.join(img_root, split, fname)
            if not os.path.isfile(fpath):
                # Buscar en todos los splits (el naming del CSV no siempre es fiable)
                for s in ('train', 'val', 'test'):
                    alt = os.path.join(img_root, s, fname)
                    if os.path.isfile(alt):
                        fpath = alt
                        break
                else:
                    continue  # imagen no encontrada, omitir

            entries.append({
                'path':   fpath,
                'label':  label,
                'split':  split,
                'source': 'plantseg',
            })

    if unmapped:
        unique_unmapped = sorted(set(unmapped))
        log.warning(f"PlantSeg: {len(unmapped)} filas sin mapeo "
                    f"({len(unique_unmapped)} combinaciones únicas) — omitidas.")
        log.debug(f"  Sin mapear: {unique_unmapped[:10]}")

    log.info(f"PlantSeg: {len(entries):,} imágenes cargadas")
    return entries


def parse_plantdec(plantdec_root: str) -> list:
    """
    PlantDEC (PlantDoc) — dataset de detección de objetos en formato YOLO.
    Cada imagen tiene un archivo .txt con líneas: class_id cx cy w h
    Una imagen puede contener múltiples IDs de clase; se usa la más frecuente.
    """
    # Localizar data.yaml para obtener el número de clases
    yaml_path = os.path.join(plantdec_root, 'data.yaml')
    if not os.path.exists(yaml_path):
        log.warning(f"data.yaml de PlantDEC no encontrado en {yaml_path}, omitiendo.")
        return []

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    yaml_classes = meta.get('names', [])
    if len(yaml_classes) != len(PLANTDEC_CLASS_MAP):
        log.warning(
            f"PlantDEC: data.yaml tiene {len(yaml_classes)} clases pero "
            f"PLANTDEC_CLASS_MAP tiene {len(PLANTDEC_CLASS_MAP)}. Verificar alineación."
        )

    entries = []
    split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}

    for split_dir, split_name in split_map.items():
        img_dir = os.path.join(plantdec_root, split_dir, 'images')
        lbl_dir = os.path.join(plantdec_root, split_dir, 'labels')
        if not os.path.isdir(img_dir):
            continue

        for fname in os.listdir(img_dir):
            if not _is_image(fname):
                continue
            img_path = os.path.join(img_dir, fname)
            lbl_path = os.path.join(lbl_dir, Path(fname).stem + '.txt')

            if not os.path.isfile(lbl_path):
                continue

            # Parsear etiqueta YOLO: elegir la clase más frecuente por imagen
            class_ids = []
            with open(lbl_path) as lf:
                for line in lf:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_ids.append(int(parts[0]))
                        except ValueError:
                            pass

            if not class_ids:
                continue

            # Clase mayoritaria para esta imagen
            majority_id = Counter(class_ids).most_common(1)[0][0]
            if majority_id >= len(PLANTDEC_CLASS_MAP):
                continue

            label = PLANTDEC_CLASS_MAP[majority_id]
            entries.append({
                'path':   img_path,
                'label':  label,
                'split':  split_name,
                'source': 'plantdec',
            })

    log.info(f"PlantDEC: {len(entries):,} imágenes cargadas")
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCTOR DEL MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════

def build_manifest(npd_root, plantseg_root, plantdec_root) -> list:
    all_entries = []

    if npd_root and os.path.isdir(npd_root):
        all_entries.extend(parse_npd(npd_root))
    else:
        log.error("Raíz de NPD no encontrada — no se puede construir el manifest sin el dataset base.")
        return []

    if plantseg_root and os.path.isdir(plantseg_root):
        all_entries.extend(parse_plantseg(plantseg_root))
    else:
        log.warning("Raíz de PlantSeg no provista o no encontrada — omitiendo.")

    if plantdec_root and os.path.isdir(plantdec_root):
        all_entries.extend(parse_plantdec(plantdec_root))
    else:
        log.warning("Raíz de PlantDEC no provista o no encontrada — omitiendo.")

    # ── Eliminar clases por debajo del mínimo de imágenes ─────────────────────
    label_counts = Counter(e['label'] for e in all_entries)
    valid_labels = {lbl for lbl, n in label_counts.items() if n >= MIN_PER_CLASS}
    dropped = sorted(set(label_counts) - valid_labels)
    if dropped:
        log.warning(
            f"Eliminando {len(dropped)} clases con <{MIN_PER_CLASS} imágenes: "
            + ', '.join(f"{d}({label_counts[d]})" for d in dropped)
        )
    all_entries = [e for e in all_entries if e['label'] in valid_labels]

    return all_entries


def write_manifest(entries: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'split', 'source'])
        writer.writeheader()
        writer.writerows(entries)
    log.info(f"Manifest escrito → {path}  ({len(entries):,} filas)")


def print_stats(entries: list):
    label_counts  = Counter(e['label']  for e in entries)
    source_counts = Counter(e['source'] for e in entries)
    split_counts  = Counter(e['split']  for e in entries)

    print()
    print("=" * 62)
    print("  ESTADÍSTICAS DEL MANIFEST")
    print("=" * 62)
    print(f"  Total de imágenes  : {len(entries):,}")
    print(f"  Clases únicas      : {len(label_counts)}")
    print()
    print("  Por fuente:")
    for src, n in sorted(source_counts.items()):
        print(f"    {src:12s}  {n:>7,} imágenes")
    print()
    print("  Por split:")
    for spl, n in sorted(split_counts.items()):
        print(f"    {spl:12s}  {n:>7,} imágenes")
    print()
    print(f"  Clases ({len(label_counts)}) con conteo de imágenes:")
    for lbl, n in sorted(label_counts.items()):
        bar = '#' * min(40, n // 50)
        print(f"    {lbl:<55s}  {n:>5,}  {bar}")
    print("=" * 62)


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE LOCALIZACIÓN DE DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_npd(base_path: str) -> str:
    for candidate in [
        os.path.join(base_path, 'New Plant Diseases Dataset(Augmented)',
                                'New Plant Diseases Dataset(Augmented)'),
        os.path.join(base_path, 'New Plant Diseases Dataset(Augmented)'),
        base_path,
    ]:
        if os.path.isdir(os.path.join(candidate, 'train')):
            return candidate
    return base_path


def download_all():
    import kagglehub
    paths = {}

    log.info("Descargando New Plant Diseases Dataset …")
    p = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    paths['npd'] = _find_npd(p)

    log.info("Descargando PlantSeg …")
    try:
        paths['plantseg'] = kagglehub.dataset_download("weitianqi/plantseg")
    except Exception as e:
        log.warning(f"Descarga de PlantSeg fallida: {e}")
        paths['plantseg'] = None

    log.info("Descargando PlantDEC …")
    try:
        paths['plantdec'] = kagglehub.dataset_download("andresmgs/plantdec")
    except Exception as e:
        log.warning(f"Descarga de PlantDEC fallida: {e}")
        paths['plantdec'] = None

    return paths


# ═══════════════════════════════════════════════════════════════════════════════
#  PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    if args.npd:
        paths = {
            'npd':      _find_npd(args.npd),
            'plantseg': args.plantseg,
            'plantdec': args.plantdec,
        }
    else:
        paths = download_all()

    entries = build_manifest(paths['npd'], paths.get('plantseg'), paths.get('plantdec'))
    if not entries:
        log.error("No se produjeron entradas — abortando.")
        return

    write_manifest(entries, MANIFEST_PATH)
    print_stats(entries)

    # Guardar lista de clases para la app y el script de entrenamiento
    class_names = sorted(set(e['label'] for e in entries))
    classes_path = os.path.join(os.path.dirname(__file__), 'model', 'class_names.json')
    os.makedirs(os.path.dirname(classes_path), exist_ok=True)
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    log.info(f"class_names.json ({len(class_names)} clases) → {classes_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Construye el manifest unificado de dataset desde NPD + PlantSeg + PlantDEC'
    )
    parser.add_argument('--npd',      default=None, help='Ruta a la raíz del dataset NPD')
    parser.add_argument('--plantseg', default=None, help='Ruta a la raíz del dataset PlantSeg')
    parser.add_argument('--plantdec', default=None, help='Ruta a la raíz del dataset PlantDEC')
    main(parser.parse_args())
