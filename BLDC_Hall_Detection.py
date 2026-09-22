import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16, ResNet50
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
import time
import random

no_delay_dir = "C:/Users/likit/BLDC_Dataset_Backup/data_in_RGB_after_preprocessing/no_delay/augmented"
fault_dir    = "C:/Users/likit/BLDC_Dataset_Backup/data_in_RGB_after_preprocessing/0.0001/augmented"

no_delay_files = [os.path.join(no_delay_dir, f) for f in os.listdir(no_delay_dir)
                   if os.path.isfile(os.path.join(no_delay_dir, f))]
fault_files_all = [os.path.join(fault_dir, f) for f in os.listdir(fault_dir)
                     if os.path.isfile(os.path.join(fault_dir, f))]

print(f"no_delay images (all): {len(no_delay_files)}, fault images (all): {len(fault_files_all)}")

# --- undersample the majority class so both classes are balanced ---
random.seed(42)
fault_files = random.sample(fault_files_all, min(len(no_delay_files), len(fault_files_all)))

print(f"Using {len(no_delay_files)} no_delay images and {len(fault_files)} fault images (balanced)")

file_paths = no_delay_files + fault_files
labels = [0] * len(no_delay_files) + [1] * len(fault_files)

combined = list(zip(file_paths, labels))
random.shuffle(combined)
file_paths, labels = zip(*combined)
file_paths, labels = list(file_paths), list(labels)

X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

num_classes = 2

def load_and_preprocess_image(file_path, label):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return img_array, label

train_data = [load_and_preprocess_image(file_path, label) for file_path, label in zip(X_train, y_train)]
X_train_processed, y_train_processed = zip(*train_data)
X_train_processed = np.array(X_train_processed)
y_train_processed = to_categorical(y_train_processed, num_classes=num_classes)

test_data = [load_and_preprocess_image(file_path, label) for file_path, label in zip(X_test, y_test)]
X_test_processed, y_test_processed = zip(*test_data)
X_test_processed = np.array(X_test_processed)
y_test_processed = to_categorical(y_test_processed, num_classes=num_classes)

def create_and_train_model(model, X_train, y_train, X_test, y_test, model_name):
    base_model = model(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f'best_{model_name}.keras', monitor='val_accuracy',
                                   save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    start_time = time.time()
    model.fit(X_train, y_train, epochs=20, batch_size=16,
              validation_data=(X_test, y_test),
              callbacks=[checkpoint, early_stop])
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    print(f'{model_name} Classification Report:')
    print(classification_report(y_test_labels, y_pred_labels, target_names=[str(i) for i in range(num_classes)]))
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'{model_name} Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
    print(f'{model_name} Parameters: {model.count_params()}')
    print(f'{model_name} Training Time: {training_time:.2f} seconds')
    print(f'{model_name} Inference Time per Sample: {inference_time:.6f} seconds')

create_and_train_model(MobileNet, X_train_processed, y_train_processed, X_test_processed, y_test_processed, 'MobileNetV1')
