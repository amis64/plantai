"""
dataset_utils.py
Carga de múltiples datasets, normalización de etiquetas y pipeline de augmentación
para clasificación de enfermedades en plantas.
"""
import os
import csv
import logging
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

IMG_SIZE  = 224
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), 'data', 'manifest.csv')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# ── Taxonomía unificada ────────────────────────────────────────────────────────
# 38 clases base del New Plant Diseases Dataset (orden alfabético)
BASE_CLASSES = [
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

# Clases extendidas encontradas frecuentemente en otros datasets de enfermedades en plantas
EXTENDED_CLASSES = [
    "Banana___Bacterial_wilt",
    "Banana___Black_sigatoka",
    "Banana___healthy",
    "Cassava___Bacterial_blight",
    "Cassava___Brown_streak_disease",
    "Cassava___Green_mottle",
    "Cassava___Mosaic_disease",
    "Cassava___healthy",
    "Coffee___Cercospora_leaf_spot",
    "Coffee___healthy",
    "Cucumber___Anthracnose",
    "Cucumber___Bacterial_wilt",
    "Cucumber___Downy_mildew",
    "Cucumber___Powdery_mildew",
    "Cucumber___healthy",
    "Mango___Anthracnose",
    "Mango___Bacterial_canker",
    "Mango___Gummosis",
    "Mango___Sooty_mould",
    "Mango___healthy",
    "Rice___Brown_spot",
    "Rice___Hispa",
    "Rice___Leaf_blast",
    "Rice___Neck_blast",
    "Rice___healthy",
    "Sugarcane___Bacterial_blight",
    "Sugarcane___Red_rot",
    "Sugarcane___Rust",
    "Sugarcane___healthy",
    "Wheat___Brown_rust",
    "Wheat___Septoria",
    "Wheat___Yellow_rust",
    "Wheat___healthy",
]

ALL_KNOWN_CLASSES = BASE_CLASSES + EXTENDED_CLASSES

# ── Normalización de etiquetas ─────────────────────────────────────────────────
def _alphanum(s: str) -> str:
    """Conserva solo caracteres alfanuméricos en minúsculas para coincidencia aproximada."""
    return ''.join(c.lower() for c in s if c.isalnum())

_NORM_TO_CANONICAL = {_alphanum(c): c for c in ALL_KNOWN_CLASSES}


def normalize_label(raw: str) -> str:
    """
    Mapea un nombre de carpeta/etiqueta crudo a la taxonomía unificada.
    Prioridad: exacto → normalizado → subcadena parcial → pasar tal cual (clase nueva).
    """
    if raw in ALL_KNOWN_CLASSES:
        return raw
    key = _alphanum(raw)
    if key in _NORM_TO_CANONICAL:
        return _NORM_TO_CANONICAL[key]
    # Coincidencia parcial por subcadena: preferir el candidato canónico más largo
    candidates = [(k, v) for k, v in _NORM_TO_CANONICAL.items()
                  if key in k or k in key]
    if candidates:
        best = max(candidates, key=lambda kv: len(kv[0]))
        return best[1]
    return raw  # Clase desconocida — se conserva tal cual y se agrega a la taxonomía


# ── Descubrimiento de dataset ──────────────────────────────────────────────────
def discover_dataset(root_dir: str):
    """
    Recorre el árbol de directorios y devuelve pares (ruta_imagen, etiqueta_carpeta).
    El nombre de la carpeta padre inmediata se usa como etiqueta de clase.
    """
    root = Path(root_dir)
    samples = []
    seen_labels: set = set()
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            label = p.parent.name
            samples.append((str(p), label))
            seen_labels.add(label)
    return samples, seen_labels


def load_dataset(root_dir: str, min_per_class: int = 10, verbose: bool = True):
    """
    Carga un dataset normalizando las etiquetas a la taxonomía unificada.

    Retorna:
        paths      : list[str]
        labels     : list[str]  (nombres canónicos unificados)
        stats      : dict
    """
    raw_samples, raw_labels = discover_dataset(root_dir)
    if not raw_samples:
        logger.warning(f"No se encontraron imágenes en {root_dir}")
        return [], [], {'total': 0, 'classes': 0}

    if verbose:
        logger.info(f"Se encontraron {len(raw_samples):,} imágenes en {len(raw_labels)} clases crudas en {root_dir}")

    paths, labels = [], []
    label_map: dict = {}
    for path, raw in raw_samples:
        if raw not in label_map:
            label_map[raw] = normalize_label(raw)
        labels.append(label_map[raw])
        paths.append(path)

    # Eliminar clases por debajo del mínimo
    counts = Counter(labels)
    valid = {cls for cls, n in counts.items() if n >= min_per_class}
    dropped = sorted(counts.keys() - valid)
    if dropped and verbose:
        logger.warning(f"Eliminando {len(dropped)} clases con <{min_per_class} muestras: {dropped[:5]}")

    pairs = [(p, l) for p, l in zip(paths, labels) if l in valid]
    if not pairs:
        return [], [], {'total': 0, 'classes': 0}
    paths_f, labels_f = zip(*pairs)

    stats = {
        'total': len(paths_f),
        'classes': len(valid),
        'dropped_classes': len(dropped),
        'label_mapping': label_map,
        'class_counts': {k: v for k, v in counts.items() if k in valid},
    }
    if verbose:
        logger.info(f"Cargadas {stats['total']:,} imágenes en {stats['classes']} clases")
    return list(paths_f), list(labels_f), stats


# ── Cargador de manifest ──────────────────────────────────────────────────────
def load_from_manifest(
    manifest_path: str = None,
    sources: list = None,      # ej. ['npd'] o ['plantseg', 'plantdec']
    splits:  list = None,      # ej. ['train'] o ['train', 'val']
) -> tuple:
    """
    Carga (rutas, etiquetas) desde el manifest CSV generado por build_dataset.py.

    Args:
        manifest_path : ruta al manifest.csv; usa data/manifest.csv por defecto
        sources       : filtrar por nombre(s) de fuente; None = todas
        splits        : filtrar por nombre(s) de split; None = todos
    Retorna:
        (paths, labels, stats)
    """
    path = manifest_path or MANIFEST_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest no encontrado en {path}. Ejecuta build_dataset.py primero."
        )

    paths, labels = [], []
    source_counts: Counter = Counter()
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if sources and row['source'] not in sources:
                continue
            if splits and row['split'] not in splits:
                continue
            if not os.path.isfile(row['path']):
                continue
            paths.append(row['path'])
            labels.append(row['label'])
            source_counts[row['source']] += 1

    counts = Counter(labels)
    stats = {
        'total':   len(paths),
        'classes': len(counts),
        'by_source': dict(source_counts),
    }
    logger.info(
        f"Manifest cargado: {stats['total']:,} imágenes  "
        f"{stats['classes']} clases  "
        f"fuentes={dict(source_counts)}"
    )
    return paths, labels, stats


# ── Pipeline tf.data ───────────────────────────────────────────────────────────
def _parse_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


# ── Funciones de augmentación ──────────────────────────────────────────────────
# Fase 1: leve — augmentaciones estándar para el dataset limpio
@tf.function
def augment_light(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    # Zoom mediante relleno + recorte aleatorio (traslación/alejamiento aleatorio)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 20, IMG_SIZE + 20)
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


# Fase 2: intensa — simular domain shift (iluminación, desenfoque, ruido, perspectiva)
@tf.function
def augment_heavy(image, label):
    # Geométrica
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Recorte de perspectiva más amplio
    pad = tf.random.uniform([], minval=30, maxval=50, dtype=tf.int32)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + pad, IMG_SIZE + pad)
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    # Variación de color
    image = tf.image.random_brightness(image, max_delta=0.25)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Simulación de desenfoque: reducir + aumentar escala, con probabilidad del 50%
    s = tf.random.uniform([], 0.5, 0.88)
    si = tf.cast(tf.cast(IMG_SIZE, tf.float32) * s, tf.int32)
    blurred = tf.image.resize(tf.image.resize(image, [si, si]), [IMG_SIZE, IMG_SIZE])
    do_blur = tf.cast(tf.random.uniform([]) > 0.5, tf.float32)
    image = do_blur * blurred + (1.0 - do_blur) * image
    # Ruido gaussiano
    noise = tf.random.normal(tf.shape(image), stddev=0.025)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image, label


# Fase 3: extrema — máxima variación para fine-tuning con datos del mundo real
@tf.function
def augment_extreme(image, label):
    # Geométrica
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Recorte de perspectiva agresivo
    pad = tf.random.uniform([], minval=50, maxval=70, dtype=tf.int32)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + pad, IMG_SIZE + pad)
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    # Variación de color agresiva
    image = tf.image.random_brightness(image, max_delta=0.35)
    image = tf.image.random_contrast(image, lower=0.4, upper=1.6)
    image = tf.image.random_saturation(image, lower=0.3, upper=1.7)
    image = tf.image.random_hue(image, max_delta=0.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Escala de grises aleatoria (simular cámara B&N o baja saturación, probabilidad 12%)
    gray = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
    do_gray = tf.cast(tf.random.uniform([]) > 0.88, tf.float32)
    image = do_gray * gray + (1.0 - do_gray) * image
    # Desenfoque intenso con probabilidad del 65%
    s = tf.random.uniform([], 0.35, 0.78)
    si = tf.cast(tf.cast(IMG_SIZE, tf.float32) * s, tf.int32)
    blurred = tf.image.resize(tf.image.resize(image, [si, si]), [IMG_SIZE, IMG_SIZE])
    do_blur = tf.cast(tf.random.uniform([]) > 0.35, tf.float32)
    image = do_blur * blurred + (1.0 - do_blur) * image
    # Ruido intenso
    noise = tf.random.normal(tf.shape(image), stddev=0.045)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    # Cutout: reemplazar un parche aleatorio con gris (simulación de oclusión)
    # Se mezcla un parche gris usando una máscara binaria suave
    patch_size = tf.random.uniform([], 20, 50, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, IMG_SIZE - patch_size, dtype=tf.int32)
    x0 = tf.random.uniform([], 0, IMG_SIZE - patch_size, dtype=tf.int32)
    # Construir máscara: 0 dentro del parche, 1 fuera
    row_idx = tf.range(IMG_SIZE)
    col_idx = tf.range(IMG_SIZE)
    row_in = tf.cast((row_idx >= y0) & (row_idx < y0 + patch_size), tf.float32)
    col_in = tf.cast((col_idx >= x0) & (col_idx < x0 + patch_size), tf.float32)
    patch_mask = tf.tensordot(row_in, col_in, axes=0)  # [H, W]
    patch_mask = tf.expand_dims(patch_mask, axis=-1)   # [H, W, 1]
    patch_mask = tf.tile(patch_mask, [1, 1, 3])         # [H, W, 3]
    # Aplicar con probabilidad del 50%
    do_cutout = tf.cast(tf.random.uniform([]) > 0.5, tf.float32)
    image = image * (1.0 - do_cutout * patch_mask) + 0.5 * do_cutout * patch_mask
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


_AUGMENT_FNS = {
    'none': None,
    'light': augment_light,
    'heavy': augment_heavy,
    'extreme': augment_extreme,
}


def build_tf_dataset(
    paths, labels, class_to_idx,
    augment: str = 'none',
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    num_classes: int = None,
) -> tf.data.Dataset:
    """
    Construye un tf.data.Dataset a partir de rutas de archivos y etiquetas en texto.

    Args:
        paths        : lista de rutas de archivos de imagen
        labels       : lista de nombres de clase unificados
        class_to_idx : dict que mapea nombre de clase → índice entero
        augment      : uno de 'none', 'light', 'heavy', 'extreme'
        batch_size   : tamaño del lote
        shuffle      : si se debe mezclar aleatoriamente
        num_classes  : si se provee, las etiquetas se codifican en one-hot (requerido
                       para CategoricalCrossentropy + label smoothing)
    """
    idx_labels = [class_to_idx.get(l, 0) for l in labels]
    ds = tf.data.Dataset.from_tensor_slices((list(paths), idx_labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 15000), seed=seed)
    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    aug_fn = _AUGMENT_FNS.get(augment)
    if aug_fn is not None:
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if num_classes is not None:
        ds = ds.map(
            lambda img, lbl: (img, tf.one_hot(lbl, num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_combined_dataset(
    dataset_parts,  # lista de tuplas (paths, labels)
    class_to_idx,
    augment: str = 'none',
    batch_size: int = 32,
    num_classes: int = None,
) -> tf.data.Dataset:
    """Combina múltiples pares (paths, labels) en un único dataset mezclado aleatoriamente."""
    all_paths, all_labels = [], []
    for paths, labels in dataset_parts:
        all_paths.extend(paths)
        all_labels.extend(labels)
    return build_tf_dataset(
        all_paths, all_labels, class_to_idx,
        augment, batch_size, shuffle=True,
        num_classes=num_classes,
    )
