# extra.py
# training_model.py

import os
import numpy as np
from collections import Counter
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# -------- CONFIG --------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATA_PATH = r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset"
MODEL_PATH = "mobilenet_mask_model_finetuned.h5"

# -------- 1. Load Data --------
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    os.path.join(r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    os.path.join(r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Validation"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    os.path.join(r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -------- 2. Compute Class Weights --------
counter = Counter(train_data.classes)
print(f"Class counts: {counter}")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
print(f"Computed class weights: {dict(enumerate(class_weights))}")

# -------- 3. Build Model --------
base_model = MobileNetV2(include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)), weights='imagenet')

# Unfreeze last 50 layers
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# -------- 4. Train Model --------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop],
    class_weight=dict(enumerate(class_weights))
)

# Save training history for plotting
np.save("training_history.npy", history.history)

# -------- 5. Evaluate and Save Model --------
loss, acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
model.save(MODEL_PATH)
