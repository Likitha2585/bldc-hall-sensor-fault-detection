import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16, ResNet50
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_pp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_pp
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_pp
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_pp
from tensorflow.keras.applications.densenet import preprocess_input as densenet_pp
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_pp
from tensorflow.keras.applications.xception import preprocess_input as xception_pp
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_pp
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
import time
import random
import csv

# ============================================================
# This is the corrected version: each model now gets ITS OWN
# correct preprocessing, instead of all 8 models being fed
# MobileNet-specific preprocessing (the bug identified after
# the first full run). Results are written to a SEPARATE file
# (results_summary_v2.csv) so your original results_summary.csv
# is never touched or overwritten -- you can compare both.
# ============================================================
QUICK_TEST = False

BASE_DATA_DIR = "C:/Users/likit/BLDC_Dataset_Backup/data_in_RGB_after_preprocessing"
NO_DELAY_DIR = f"{BASE_DATA_DIR}/no_delay/augmented"

CONDITIONS = ["0.0001", "0.005", "0.01"]

# Each entry: (model class, name, its correct preprocess_input function)
ALL_MODEL_CONFIGS = [
    (MobileNet, "MobileNetV1", mobilenet_pp),
    (MobileNetV2, "MobileNetV2", mobilenetv2_pp),
    (VGG16, "VGG16", vgg16_pp),
    (ResNet50, "ResNet50", resnet50_pp),
    (DenseNet121, "DenseNet121", densenet_pp),
    (InceptionV3, "InceptionV3", inceptionv3_pp),
    (Xception, "Xception", xception_pp),
    (EfficientNetB0, "EfficientNetB0", efficientnet_pp),
]

if QUICK_TEST:
    CONDITIONS = CONDITIONS[:1]
    # For the quick test, specifically check one of the previously-failing
    # models (ResNet50) rather than MobileNet again, since that's what
    # we're actually trying to verify this time.
    MODEL_CONFIGS = [c for c in ALL_MODEL_CONFIGS if c[1] == "ResNet50"]
    print(">>> QUICK_TEST is ON: running only 1 condition x ResNet50 to verify the preprocessing fix.")
else:
    MODEL_CONFIGS = ALL_MODEL_CONFIGS
    print(">>> QUICK_TEST is OFF: running the FULL 3-condition x 8-model comparison with corrected preprocessing.")

num_classes = 2
random.seed(42)

no_delay_files = [os.path.join(NO_DELAY_DIR, f) for f in os.listdir(NO_DELAY_DIR)
                   if os.path.isfile(os.path.join(NO_DELAY_DIR, f))]

def load_raw_image(file_path, label):
    """Loads image WITHOUT any preprocessing applied -- just resized,
    raw 0-255 pixel values. Model-specific preprocessing is applied
    later, per model, right before training."""
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    return img_array, label

def build_dataset_for_condition(condition):
    fault_dir = f"{BASE_DATA_DIR}/{condition}/augmented"
    fault_files_all = [os.path.join(fault_dir, f) for f in os.listdir(fault_dir)
                         if os.path.isfile(os.path.join(fault_dir, f))]
    fault_files = random.sample(fault_files_all, min(len(no_delay_files), len(fault_files_all)))

    print(f"[{condition}] no_delay: {len(no_delay_files)}, fault (undersampled): {len(fault_files)}")

    file_paths = no_delay_files + fault_files
    labels = [0] * len(no_delay_files) + [1] * len(fault_files)

    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths, labels = zip(*combined)
    file_paths, labels = list(file_paths), list(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_data = [load_raw_image(fp, l) for fp, l in zip(X_train, y_train)]
    X_train_raw, y_train_raw = zip(*train_data)
    X_train_raw = np.array(X_train_raw)
    y_train_p = to_categorical(y_train_raw, num_classes=num_classes)

    test_data = [load_raw_image(fp, l) for fp, l in zip(X_test, y_test)]
    X_test_raw, y_test_raw = zip(*test_data)
    X_test_raw = np.array(X_test_raw)
    y_test_p = to_categorical(y_test_raw, num_classes=num_classes)

    return X_train_raw, y_train_p, X_test_raw, y_test_p

def build_model(base_model_fn, model_name):
    data_augmentation = Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ], name=f"{model_name}_augmentation")

    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(base_model_fn, model_name, preprocess_fn, condition,
                        X_train_raw, y_train, X_test_raw, y_test, results):
    tag = f"{model_name}_{condition}"
    print(f"\n=== Training {model_name} on condition {condition} (correct preprocessing) ===")

    # Apply THIS model's correct preprocessing to a copy of the raw images
    X_train = preprocess_fn(X_train_raw.copy())
    X_test = preprocess_fn(X_test_raw.copy())

    model = build_model(base_model_fn, model_name)

    checkpoint = ModelCheckpoint(f'best_{tag}_v2.keras', monitor='val_accuracy',
                                   save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    start_time = time.time()
    model.fit(X_train, y_train, epochs=20, batch_size=16,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stop], verbose=1)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    report = classification_report(y_test_labels, y_pred_labels,
                                     target_names=['no_delay', 'fault'], output_dict=True, zero_division=0)
    print(f'{tag} Classification Report:')
    print(classification_report(y_test_labels, y_pred_labels, target_names=['no_delay', 'fault'], zero_division=0))

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'{tag} Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    results.append({
        "condition": condition,
        "model": model_name,
        "test_accuracy": round(accuracy, 4),
        "test_loss": round(loss, 4),
        "precision_no_delay": round(report['no_delay']['precision'], 4),
        "recall_no_delay": round(report['no_delay']['recall'], 4),
        "precision_fault": round(report['fault']['precision'], 4),
        "recall_fault": round(report['fault']['recall'], 4),
        "params": model.count_params(),
        "training_time_sec": round(training_time, 2),
        "inference_time_sec": round(inference_time, 6),
    })

results = []

for condition in CONDITIONS:
    X_train_raw, y_train, X_test_raw, y_test = build_dataset_for_condition(condition)
    for base_model_fn, model_name, preprocess_fn in MODEL_CONFIGS:
        train_and_evaluate(base_model_fn, model_name, preprocess_fn, condition,
                            X_train_raw, y_train, X_test_raw, y_test, results)

csv_path = "results_summary_v2.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"\nAll done. Summary written to {csv_path}")
print("Compare this against results_summary.csv to see if the fix helped.")
for r in results:
    print(r)
