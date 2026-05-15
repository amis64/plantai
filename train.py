"""
Training script for the Plant Disease Classifier.
Uses MobileNetV2 with transfer learning in two phases.

Usage:
    python train.py                          # Download dataset from Kaggle automatically
    python train.py <path_to_dataset>        # Use local dataset path
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 38
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_model.keras')


def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs, outputs), base_model


def get_generators(train_dir, valid_dir):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    valid_gen = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, valid_gen


def train(dataset_root):
    train_dir = os.path.join(dataset_root, 'train')
    valid_dir = os.path.join(dataset_root, 'valid')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

    train_gen, valid_gen = get_generators(train_dir, valid_dir)
    print(f"Train: {train_gen.samples} images | Valid: {valid_gen.samples} images")
    print(f"Classes: {train_gen.num_classes}")

    model, base_model = build_model()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Phase 1: Train head only (base frozen)
    print("\n=== Phase 1: Training classification head ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        epochs=20,
        validation_data=valid_gen,
        callbacks=callbacks
    )

    # Phase 2: Fine-tune last 30 layers of MobileNetV2
    print("\n=== Phase 2: Fine-tuning last 30 layers ===")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        epochs=10,
        validation_data=valid_gen,
        callbacks=callbacks
    )

    print(f"\nModel saved to: {MODEL_PATH}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    loss, acc = model.evaluate(valid_gen, verbose=1)
    print(f"Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Validation Loss: {loss:.4f}")

    return model


def download_and_prepare_dataset():
    """Download dataset from Kaggle using kagglehub."""
    try:
        import kagglehub
        print("Downloading New Plant Diseases Dataset from Kaggle...")
        path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
        dataset_root = os.path.join(
            path,
            'New Plant Diseases Dataset(Augmented)',
            'New Plant Diseases Dataset(Augmented)'
        )
        if not os.path.exists(dataset_root):
            # Try alternate path structure
            dataset_root = path
        return dataset_root
    except ImportError:
        print("kagglehub not installed. Install with: pip install kagglehub")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Download the dataset manually from:")
        print("  https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("Then run: python train.py <path_to_extracted_dataset>")
        sys.exit(1)


if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
        print(f"Using dataset at: {dataset_root}")
    else:
        dataset_root = download_and_prepare_dataset()
        print(f"Dataset at: {dataset_root}")

    train(dataset_root)
