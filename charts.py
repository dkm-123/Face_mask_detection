# classification_report.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # ✅ IMPORTANT!
from sklearn.metrics import classification_report

# --- CONFIG ---
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "mobilenet_mask_model_finetuned.h5"
TEST_DIR = r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection\Face Mask Dataset\Test"

# --- LOAD MODEL ---
model = load_model(MODEL_PATH)

# --- LOAD TEST DATA ---
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)  # ✅ USE CORRECT PREPROCESSING

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- PREDICT ---
y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# --- CHECK PREDICTIONS ---
print(f"Unique predicted labels: {np.unique(y_pred)}")
print(f"Counts: {np.bincount(y_pred)}")

# --- REPORT ---
report = classification_report(
    y_true,
    y_pred,
    target_names=['Mask', 'No Mask'],
    output_dict=True,
    zero_division=0  # prevents undefined metric warnings
)

print("\nClassification Report (raw):")
print(report)

# --- BAR CHART ---
labels = ['Mask', 'No Mask']
metrics = ['precision', 'recall', 'f1-score']

# Safely get metric values, default to 0.0 if missing
data = []
for metric in metrics:
    values = []
    for cls in labels:
        value = report.get(cls, {}).get(metric, 0.0)
        values.append(value)
    data.append(values)

x = np.arange(len(labels))
bar_width = 0.2

plt.figure(figsize=(8, 5))
for i, metric in enumerate(metrics):
    plt.bar(x + i * bar_width, data[i], width=bar_width, label=metric)

plt.xticks(x + bar_width, labels)
plt.ylim(0, 1)
plt.title("Classification Report - Mask vs No Mask")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()

plt.savefig("classification_report.png")
plt.show()

print("\n✅ Bar chart saved as 'classification_report.png' in the same folder.")