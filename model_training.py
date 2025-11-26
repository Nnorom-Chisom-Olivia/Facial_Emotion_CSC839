import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Image size for FER2013
IMG_SIZE = 48

# Image data generators (loads images directly from folders)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# Build CNN model
model = Sequential([
    Conv2D(64, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # FER2013 has 7 emotions
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save the best model during training
checkpoint = ModelCheckpoint(
    "face_emotionModel.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# Train the model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=30,
    callbacks=[checkpoint]
)

print("Training complete. Model saved as face_emotionModel.h5")