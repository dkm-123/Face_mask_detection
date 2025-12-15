# prediction_confidence.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIG ---
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "mobilenet_mask_model_finetuned.h5"
TEST_DIR = r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Test"

# --- LOAD MODEL & DATA ---
model = load_model(MODEL_PATH)

test_gen = ImageDataGenerator(preprocessing_function=None)
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- PREDICT PROBABILITIES ---
y_true = test_data.classes
y_pred_probs = model.predict(test_data).flatten()

# --- HISTOGRAM ---
plt.figure(figsize=(8,5))
plt.hist(y_pred_probs[y_true == 0], bins=20, alpha=0.7, label='Mask')
plt.hist(y_pred_probs[y_true == 1], bins=20, alpha=0.7, label='No Mask')

plt.xlabel("Predicted Probability of 'No Mask'")
plt.ylabel("Number of Images")
plt.title("Prediction Confidence Histogram")
plt.legend()
plt.tight_layout()

plt.savefig("prediction_confidence_histogram.png")
plt.show()

print("\n✅ Saved as prediction_confidence_histogram.png")