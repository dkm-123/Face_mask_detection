# data_preprocessing.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIG ---
IMG_SIZE, BATCH_SIZE = 224, 32
BASE_PATH = r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset"

# --- Data Generators ---
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
val_test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_gen.flow_from_directory(
    os.path.join(r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val = val_test_gen.flow_from_directory(
    os.path.join(r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Validation"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test = val_test_gen.flow_from_directory(
    os.path.join(r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- Output ---
print("Classes:", train.class_indices)
print(f"Train: {train.samples} | Val: {val.samples} | Test: {test.samples}")
