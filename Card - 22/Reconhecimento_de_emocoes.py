import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data():
    data = pd.read_csv('./fer2013.csv')
    pixels = data['pixels'].tolist()
    faces = np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(48, 48) for pixel in pixels])
    emotions = to_categorical(data['emotion'], num_classes=7)
    faces = faces.astype('float32') / 255.0
    faces = np.expand_dims(faces, -1)
    return train_test_split(faces, emotions, test_size=0.2, random_state=42)

def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

x_train, x_test, y_train, y_test = load_data()
model = build_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint('emotion_model.keras', save_best_only=True, monitor='val_loss')
]

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=100, callbacks=callbacks)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}') 
