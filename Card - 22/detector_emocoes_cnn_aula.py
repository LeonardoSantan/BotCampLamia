import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Carrega o dataset de emoções faciais
print("Carregando o dataset...")
data = pd.read_csv('./Material/fer2013/fer2013/fer2013.csv')

# Processa as imagens do dataset
pixels = data['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    faces.append(face)

faces = np.expand_dims(np.array(faces), axis=-1)
faces = faces.astype('float32') / 255.0

# Prepara os rótulos em formato one-hot
emotions = pd.get_dummies(data['emotion'], dtype=int).values

# Divide os dados em conjuntos de treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

# Salva os conjuntos de teste para uso futuro
np.save('mod_xtest.npy', X_test)
np.save('mod_ytest.npy', y_test)

# Define a arquitetura da rede neural convolucional
print("Configurando o modelo...")
num_features = 64
num_labels = 7
model = Sequential([
    Input((48, 48, 1)),
    Conv2D(num_features, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Conv2D(2 * num_features, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_labels, activation='softmax')
])

# Compila o modelo com otimizador Adam e função de perda
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Configura callbacks para ajuste dinâmico do treinamento
model_file = 'model_emotions.h5'
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, verbose=1),
    ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=1)
]

# Inicia o treinamento da rede neural
print("Iniciando o treinamento...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Avalia o modelo no conjunto de teste
evaluation = model.evaluate(X_test, y_test)
print(f"Resultados - Perda: {evaluation[0]:.4f}, Acurácia: {evaluation[1]:.4f}")

# Carrega uma imagem para demonstração de detecção de emoções
print("Carregando imagem de teste...")
image = cv2.imread('./Material/testes/teste03.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.axis('off')
plt.show()

# Detecta faces na imagem e prevê as emoções
face_cascade = cv2.CascadeClassifier('./Material/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

emotions_labels = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48)).astype('float32') / 255.0
    cropped_img = np.expand_dims(np.expand_dims(roi_gray, axis=-1), axis=0)
    prediction = model.predict(cropped_img)[0]
    emotion = emotions_labels[np.argmax(prediction)]
    cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

# Exibe a imagem processada com as emoções detectadas
plt.imshow(image)
plt.axis('off')
plt.show()
