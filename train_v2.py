

import os
import sys
import json
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from collections import Counter

from dataset_utils import (
    load_dataset, load_from_manifest,
    build_tf_dataset, build_combined_dataset,
    MANIFEST_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

IMG_SIZE     = 224
MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH   = os.path.join(MODEL_DIR, 'plant_disease_model.keras')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.json')


# ── Descarga de datasets ──────────────────────────────────────────────────────
def download_datasets() -> dict:
    """Descarga los tres datasets de Kaggle usando kagglehub."""
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub no está instalado. Ejecuta: pip install kagglehub")
        sys.exit(1)

    datasets = {}

    # New Plant Diseases Dataset
    logger.info("Descargando New Plant Diseases Dataset …")
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    # Localizar la raíz real de imágenes (el dataset tiene directorios anidados idénticos)
    for candidate in [
        os.path.join(path, 'New Plant Diseases Dataset(Augmented)',
                           'New Plant Diseases Dataset(Augmented)'),
        os.path.join(path, 'New Plant Diseases Dataset(Augmented)'),
        path,
    ]:
        if os.path.isdir(candidate):
            datasets['npd'] = candidate
            break
    logger.info(f"Raíz NPD: {datasets.get('npd')}")

    # PlantSeg
    logger.info("Descargando dataset PlantSeg …")
    try:
        path = kagglehub.dataset_download("weitianqi/plantseg")
        datasets['plantseg'] = path
        logger.info(f"Raíz PlantSeg: {path}")
    except Exception as e:
        logger.warning(f"Descarga de PlantSeg fallida ({e}). Continuando sin él.")
        datasets['plantseg'] = None

    # PlantDEC
    logger.info("Descargando dataset PlantDEC …")
    try:
        path = kagglehub.dataset_download("andresmgs/plantdec")
        datasets['plantdec'] = path
        logger.info(f"Raíz PlantDEC: {path}")
    except Exception as e:
        logger.warning(f"Descarga de PlantDEC fallida ({e}). Continuando sin él.")
        datasets['plantdec'] = None

    return datasets


# ── Arquitectura del modelo ───────────────────────────────────────────────────
def build_model(num_classes: int, backbone: str = 'mobilenetv2', dropout: float = 0.4):
    """
    Construye el clasificador sobre un backbone preentrenado.

    Mejoras respecto a v1:
    - BatchNormalization antes del dropout para entrenamiento estable
    - Capa Dense adicional con regularización L2 para mayor capacidad
    - Dropout más alto para contrarrestar la cabeza más grande
    - Opción EfficientNetB0 para representaciones de características más potentes
    """
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')

    if backbone == 'efficientnet':
        base = EfficientNetB0(include_top=False, weights='imagenet')
        # EfficientNet espera rango [0, 255]; nuestro pipeline produce [0, 1]
        x = layers.Lambda(lambda t: t * 255.0, name='scale_255')(inputs)
        x = base(x, training=False)
    else:
        base = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet',
        )
        x = base(inputs, training=False)

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(dropout, name='drop1')(x)
    x = layers.Dense(
        512, activation='relu',
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
        name='dense_head',
    )(x)
    x = layers.Dropout(dropout * 0.6, name='drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def set_base_trainable(base, n_trainable_tail: int = 0):
    """Congela todas las capas del backbone excepto las últimas n_trainable_tail."""
    if n_trainable_tail == 0:
        base.trainable = False
        return
    base.trainable = True
    cutoff = len(base.layers) - n_trainable_tail
    for layer in base.layers[:cutoff]:
        layer.trainable = False
    logger.info(
        f"Base: {n_trainable_tail}/{len(base.layers)} capas descongeladas "
        f"(parámetros entrenables ≈ {sum(int(np.prod(v.shape)) for v in base.trainable_weights):,})"
    )


def compile_model(model, lr: float, label_smoothing: float = 0.1):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc'),
        ],
    )


# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(patience_stop: int = 7, patience_lr: int = 3):
    return [
        EarlyStopping(
            monitor='val_accuracy', patience=patience_stop,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience_lr, min_lr=1e-7, verbose=1,
        ),
    ]


# ── Evaluación rápida ─────────────────────────────────────────────────────────
def quick_evaluate(model, val_ds, class_names: list):
    """Evaluación rápida con sklearn; se llama al final de cada fase."""
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        logger.warning("scikit-learn no instalado — omitiendo evaluación F1.")
        return

    all_pred, all_true = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        all_pred.extend(np.argmax(preds, axis=1))
        lbl = labels.numpy()
        # Las etiquetas son one-hot (forma [B, C]) — convertir de vuelta a índices enteros
        if lbl.ndim == 2:
            lbl = np.argmax(lbl, axis=1)
        all_true.extend(lbl)

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)

    acc       = np.mean(all_pred == all_true)
    f1_macro  = f1_score(all_true, all_pred, average='macro',    zero_division=0)
    f1_weight = f1_score(all_true, all_pred, average='weighted', zero_division=0)
    logger.info(
        f"  Val accuracy={acc:.4f} | F1-macro={f1_macro:.4f} | F1-weighted={f1_weight:.4f}"
    )
    return {'accuracy': float(acc), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_weight)}


# ── Principal ─────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Construir o localizar el manifest ────────────────────────────────────
    if not os.path.exists(MANIFEST_PATH):
        logger.info("Manifest no encontrado — ejecutando build_dataset.py primero …")
        from build_dataset import build_manifest, write_manifest, download_all, _find_npd

        if args.dataset_npd:
            paths_dict = {
                'npd':      _find_npd(args.dataset_npd),
                'plantseg': args.dataset_plantseg,
                'plantdec': args.dataset_plantdec,
            }
        else:
            paths_dict = download_all()

        entries = build_manifest(
            paths_dict['npd'],
            paths_dict.get('plantseg'),
            paths_dict.get('plantdec'),
        )
        if not entries:
            logger.error("La construcción del manifest no produjo entradas. Abortando.")
            sys.exit(1)
        write_manifest(entries, MANIFEST_PATH)
    else:
        logger.info(f"Usando manifest existente: {MANIFEST_PATH}")

    # ── Cargar desde el manifest ──────────────────────────────────────────────
    logger.info("\n=== Cargando datasets desde el manifest ===")

    # Fuente de la Fase 1: imágenes limpias de NPD (solo split train)
    npd_paths, npd_labels, npd_stats = load_from_manifest(
        sources=['npd'], splits=['train']
    )
    logger.info(f"NPD train: {npd_stats['total']:,} imágenes  {npd_stats['classes']} clases")

    # Fuentes del mundo real: PlantSeg + PlantDEC (split train)
    real_paths, real_labels, real_stats = load_from_manifest(
        sources=['plantseg', 'plantdec'], splits=['train']
    )
    has_real = real_stats['total'] > 0
    if has_real:
        logger.info(
            f"Mundo real: {real_stats['total']:,} imágenes  "
            f"{real_stats['classes']} clases  {real_stats['by_source']}"
        )

    # ── Construir taxonomía unificada de clases ───────────────────────────────
    all_labels = npd_labels + real_labels
    class_counts = Counter(all_labels)
    class_names = sorted(class_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)

    logger.info(f"\nTaxonomía unificada: {num_classes} clases en total")
    logger.info(f"  Clases de New Plant Diseases: {len(set(npd_labels))}")
    if has_real:
        logger.info(f"  Clases de datasets del mundo real: {len(set(real_labels))}")

    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    logger.info(f"Nombres de clases guardados → {CLASSES_PATH}")

    # ── Construir conjunto de validación ──────────────────────────────────────
    # Usar el split val de NPD en el manifest; si no existe, usar 10% del train de NPD
    try:
        val_paths_f, val_labels_f, _ = load_from_manifest(
            sources=['npd'], splits=['val']
        )
        # Conservar solo clases que existen en la taxonomía de entrenamiento
        pairs = [(p, l) for p, l in zip(val_paths_f, val_labels_f) if l in class_to_idx]
        val_paths_f = [p for p, l in pairs]
        val_labels_f = [l for p, l in pairs]
    except Exception:
        val_paths_f, val_labels_f = [], []

    if not val_paths_f:
        logger.info("No hay split val en el manifest — usando 10% del train de NPD como validación.")
        n = len(npd_paths)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        val_n = max(1, int(n * 0.10))
        val_idx, train_idx = idx[:val_n], idx[val_n:]
        val_paths_f  = [npd_paths[i]  for i in val_idx]
        val_labels_f = [npd_labels[i] for i in val_idx]
        npd_paths    = [npd_paths[i]  for i in train_idx]
        npd_labels   = [npd_labels[i] for i in train_idx]
    val_ds = build_tf_dataset(
        list(val_paths_f), list(val_labels_f),
        class_to_idx, augment='none',
        batch_size=args.batch_size, shuffle=False,
        num_classes=num_classes,
    )
    logger.info(f"Conjunto de validación: {len(val_paths_f):,} imágenes")

    # ── Construir modelo ──────────────────────────────────────────────────────
    logger.info(f"\nConstruyendo modelo — backbone={args.backbone}  clases={num_classes}")
    model, base = build_model(num_classes, backbone=args.backbone)
    total_params = model.count_params()
    logger.info(f"Total de parámetros: {total_params:,}")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1 — Dataset limpio, backbone congelado, augmentación leve
    # Objetivo: aprender características de enfermedades en imágenes de laboratorio
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("FASE 1 — Dataset limpio | base congelada | augmentación leve")
    logger.info("="*60)
    set_base_trainable(base, n_trainable_tail=0)
    compile_model(model, lr=1e-3, label_smoothing=0.1)

    train_ds_p1 = build_tf_dataset(
        npd_paths, npd_labels, class_to_idx,
        augment='light', batch_size=args.batch_size,
        num_classes=num_classes,
    )
    model.fit(
        train_ds_p1,
        epochs=args.phase1_epochs,
        validation_data=val_ds,
        callbacks=make_callbacks(patience_stop=7, patience_lr=3),
    )
    logger.info("Fase 1 completada.")
    quick_evaluate(model, val_ds, class_names)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — Dataset mixto, descongelamiento parcial, augmentación intensa
    # Objetivo: adaptarse al domain shift conservando el conocimiento del dataset limpio
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("FASE 2 — Dataset mixto | descongelamiento parcial | augmentación intensa")
    logger.info("="*60)

    # Descongelar más capas para EfficientNet (más profundo = más parámetros)
    n_tail_p2 = 40 if args.backbone == 'efficientnet' else 30
    set_base_trainable(base, n_trainable_tail=n_tail_p2)
    compile_model(model, lr=5e-5, label_smoothing=0.1)

    if has_real:
        # Mezcla balanceada: igual número de imágenes limpias y reales
        n_real_sample = min(len(real_paths), len(npd_paths))
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(real_paths), n_real_sample, replace=False)
        s_real_paths  = [real_paths[i]  for i in sample_idx]
        s_real_labels = [real_labels[i] for i in sample_idx]
        mixed_parts = [(npd_paths, npd_labels), (s_real_paths, s_real_labels)]
    else:
        mixed_parts = [(npd_paths, npd_labels)]

    train_ds_p2 = build_combined_dataset(
        mixed_parts, class_to_idx,
        augment='heavy', batch_size=args.batch_size,
        num_classes=num_classes,
    )
    model.fit(
        train_ds_p2,
        epochs=args.phase2_epochs,
        validation_data=val_ds,
        callbacks=make_callbacks(patience_stop=6, patience_lr=3),
    )
    logger.info("Fase 2 completada.")
    quick_evaluate(model, val_ds, class_names)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3 — Fine-tuning con datos del mundo real, descongelamiento profundo, augmentación extrema
    # Objetivo: maximizar la robustez en imágenes reales (campo, celular, iluminación variable)
    # ══════════════════════════════════════════════════════════════════════════
    if has_real:
        logger.info("\n" + "="*60)
        logger.info("FASE 3 — Solo mundo real | descongelamiento profundo | augmentación extrema")
        logger.info("="*60)

        n_tail_p3 = 80 if args.backbone == 'efficientnet' else 60
        set_base_trainable(base, n_trainable_tail=n_tail_p3)
        compile_model(model, lr=1e-5, label_smoothing=0.15)

        train_ds_p3 = build_tf_dataset(
            real_paths, real_labels, class_to_idx,
            augment='extreme', batch_size=args.batch_size,
            num_classes=num_classes,
        )
        model.fit(
            train_ds_p3,
            epochs=args.phase3_epochs,
            validation_data=val_ds,
            callbacks=make_callbacks(patience_stop=5, patience_lr=2),
        )
        logger.info("Fase 3 completada.")
    else:
        logger.warning("No se cargaron datasets del mundo real — Fase 3 omitida.")
        logger.warning("  → El modelo igual generalizará mejor que v1 gracias a la")
        logger.warning("    augmentación intensa de la Fase 2 y el backbone mejorado.")

    # ── Evaluación final ──────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("EVALUACIÓN FINAL")
    logger.info("="*60)
    metrics = quick_evaluate(model, val_ds, class_names) or {}

    with open(os.path.join(MODEL_DIR, 'train_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nModelo  → {MODEL_PATH}")
    logger.info(f"Clases  → {CLASSES_PATH}")
    logger.info("Listo. Ejecuta evaluate.py para el informe completo de clasificación + matriz de confusión.")

    return model


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenamiento multi-dataset con curriculum learning para clasificación de enfermedades en plantas'
    )
    parser.add_argument(
        '--backbone', choices=['mobilenetv2', 'efficientnet'], default='mobilenetv2',
        help='Backbone: mobilenetv2 (por defecto, más rápido) o efficientnet (mayor precisión)',
    )
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--phase1_epochs',  type=int, default=20,
                        help='Épocas para la Fase 1 (limpio, base congelada)')
    parser.add_argument('--phase2_epochs',  type=int, default=15,
                        help='Épocas para la Fase 2 (mixto, descongelamiento parcial)')
    parser.add_argument('--phase3_epochs',  type=int, default=10,
                        help='Épocas para la Fase 3 (fine-tuning con datos reales)')
    parser.add_argument('--dataset_npd',      type=str, default=None,
                        help='Ruta local a la raíz del dataset New Plant Diseases')
    parser.add_argument('--dataset_plantseg', type=str, default=None,
                        help='Ruta local a la raíz del dataset PlantSeg')
    parser.add_argument('--dataset_plantdec', type=str, default=None,
                        help='Ruta local a la raíz del dataset PlantDEC')
    args = parser.parse_args()
    main(args)
