"""
evaluate.py
Evaluación completa del modelo: accuracy, top-5, F1 (macro + ponderado),
reporte por clase y matriz de confusión.

Uso:
    python evaluate.py --val_dir <ruta_a_imágenes_de_validación>
    python evaluate.py --val_dir <ruta> --model_path model/plant_disease_model.keras
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH   = os.path.join(MODEL_DIR, 'plant_disease_model.keras')
CLASSES_PATH = os.path.join(MODEL_DIR, 'class_names.json')


def load_class_names() -> list:
    """Carga los nombres de clase guardados por train_v2.py; usa la lista de app.py como respaldo."""
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, encoding='utf-8') as f:
            return json.load(f)
    # Respaldo: importar desde app.py
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'app', os.path.join(os.path.dirname(__file__), 'app.py')
        )
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)
        return app_mod.CLASS_NAMES
    except Exception:
        raise RuntimeError(
            f"class_names.json no encontrado en {CLASSES_PATH} y la importación de app.py falló. "
            "Ejecuta train_v2.py primero."
        )


def evaluate(model_path: str, val_dir: str, batch_size: int = 32):
    from dataset_utils import load_dataset, build_tf_dataset

    # Cargar modelo
    print(f"Cargando modelo desde {model_path} …")
    model = tf.keras.models.load_model(model_path)

    class_names = load_class_names()
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Cargar datos de validación
    print(f"Cargando datos de validación desde {val_dir} …")
    paths, labels, stats = load_dataset(val_dir, min_per_class=1, verbose=True)
    if not paths:
        print("No se encontraron imágenes. Verifica --val_dir.")
        return

    # Filtrar a las clases presentes en el modelo entrenado
    pairs = [(p, l) for p, l in zip(paths, labels) if l in class_to_idx]
    skipped = len(paths) - len(pairs)
    if skipped:
        print(f"  Se omitieron {skipped} imágenes con clases que no están en el modelo entrenado.")
    if not pairs:
        print("No se encontraron clases coincidentes entre val_dir y el modelo. Abortando.")
        return

    paths_f, labels_f = zip(*pairs)
    val_ds = build_tf_dataset(
        list(paths_f), list(labels_f),
        class_to_idx, augment='none',
        batch_size=batch_size, shuffle=False,
    )

    # Recolectar predicciones
    print("Ejecutando inferencia …")
    all_proba, all_true = [], []
    for images, label_batch in val_ds:
        proba = model.predict(images, verbose=0)
        all_proba.extend(proba)
        all_true.extend(label_batch.numpy())

    all_proba = np.array(all_proba)
    all_true  = np.array(all_true,  dtype=int)
    all_pred  = np.argmax(all_proba, axis=1)

    # ── Métricas ──────────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix, f1_score,
            top_k_accuracy_score,
        )
    except ImportError:
        print("scikit-learn es requerido. Ejecuta: pip install scikit-learn")
        return

    acc        = float(np.mean(all_pred == all_true))
    top5_k     = min(5, len(class_names))
    top5_acc   = float(top_k_accuracy_score(all_true, all_proba, k=top5_k))
    f1_macro   = float(f1_score(all_true, all_pred, average='macro',    zero_division=0))
    f1_weight  = float(f1_score(all_true, all_pred, average='weighted', zero_division=0))

    present = sorted(set(all_true.tolist()))
    pnames  = [class_names[i] for i in present]

    report = classification_report(
        all_true, all_pred,
        labels=present, target_names=pnames,
        zero_division=0,
    )

    # ── Imprimir resultados ───────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("  RESULTADOS DE EVALUACIÓN")
    print(sep)
    print(f"  Muestras evaluadas  : {len(all_true):,}")
    print(f"  Clases presentes    : {len(present)}")
    print(f"  Accuracy Top-1      : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Accuracy Top-{top5_k}      : {top5_acc:.4f}  ({top5_acc*100:.2f}%)")
    print(f"  F1 Macro            : {f1_macro:.4f}")
    print(f"  F1 Ponderado        : {f1_weight:.4f}")
    print(sep)
    print(f"\nReporte por clase:\n{report}")

    # ── Guardar reporte ───────────────────────────────────────────────────────
    metrics = {
        'accuracy':     acc,
        'top5_accuracy': top5_acc,
        'f1_macro':     f1_macro,
        'f1_weighted':  f1_weight,
    }
    report_txt = os.path.join(MODEL_DIR, 'eval_report.txt')
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write(f"Accuracy Top-1  : {acc:.4f}\n")
        f.write(f"Accuracy Top-5  : {top5_acc:.4f}\n")
        f.write(f"F1 Macro        : {f1_macro:.4f}\n")
        f.write(f"F1 Ponderado    : {f1_weight:.4f}\n\n")
        f.write(report)
    print(f"Reporte guardado → {report_txt}")

    metrics_json = os.path.join(MODEL_DIR, 'eval_metrics.json')
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas JSON → {metrics_json}")

    # ── Matriz de confusión ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(all_true, all_pred, labels=present)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)

        fig_size = max(14, len(present) // 2)
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

        annot = len(present) <= 20
        short_names = [n.split('___')[-1][:16] for n in pnames]
        sns.heatmap(
            cm_norm, annot=annot, fmt='.2f', cmap='Blues',
            xticklabels=short_names, yticklabels=short_names,
            ax=ax, vmin=0, vmax=1,
        )
        ax.set_xlabel('Predicho', fontsize=12)
        ax.set_ylabel('Etiqueta real', fontsize=12)
        ax.set_title('Matriz de confusión normalizada', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"Matriz de confusión → {cm_path}")
        plt.close()

    except ImportError:
        print("matplotlib/seaborn no instalados — matriz de confusión omitida.")
        print("  Instala con: pip install matplotlib seaborn")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evalúa el modelo de enfermedades en plantas con F1, top-5 y matriz de confusión'
    )
    parser.add_argument(
        '--val_dir', required=True,
        help='Ruta al directorio de imágenes de validación (subcarpetas por clase)',
    )
    parser.add_argument(
        '--model_path', default=MODEL_PATH,
        help=f'Ruta al modelo guardado (por defecto: {MODEL_PATH})',
    )
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    evaluate(args.model_path, args.val_dir, args.batch_size)
