import os
import cv2
import numpy as np
import winsound
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -------- CONFIG --------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
DATA_PATH = r"C:\Users\Admin\OneDrive\Desktop\face_mask_detection\dataset_for_maskdetection"
MODEL_PATH = "mobilenet_mask_model_finetuned.h5"
HISTORY_PATH = "training_history.npy"

# -------- 1. Load Data --------
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    os.path.join(DATA_PATH, "Face Mask Dataset", "Train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    os.path.join(DATA_PATH, "Face Mask Dataset", "Validation"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    os.path.join(DATA_PATH, "Face Mask Dataset", "Test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# -------- 2. Build Model --------
base_model = MobileNetV2(include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)), weights='imagenet')

# Freeze all layers except last 20
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# -------- 3. Train Model --------
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -------- 4. Save Training History (.npy) --------
if history.history:
    np.save(HISTORY_PATH, history.history)
    print(f"\n✅ Training history saved to {HISTORY_PATH}")
else:
    print("\n❌ Training history is empty. Not saved.")

# -------- 5. Evaluate and Save Model --------
loss, acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
model.save(MODEL_PATH)

# -------- 6. Load Model and Haar Cascade --------
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------- 7. Real-Time Detection with Beep --------
cap = cv2.VideoCapture(0)
print("\n🎥 Webcam started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face)[0][0]

            if pred < 0.5:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
                winsound.Beep(1000, 300)  # 🔊 Beep for No Mask

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        except:
            continue

    cv2.imshow("Face Mask Detection (with Beep)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
