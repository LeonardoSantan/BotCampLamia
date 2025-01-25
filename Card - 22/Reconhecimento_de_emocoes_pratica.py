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

# Função para carregar e pré-processar os dados
def load_data():
    """
    Carrega o dataset FER-2013 (arquivo CSV) e realiza o pré-processamento das imagens e rótulos de emoção.
    As imagens são convertidas de strings para arrays NumPy e normalizadas para o intervalo [0, 1].
    A codificação das emoções é feita em one-hot encoding.
    
    Retorna:
        x_train, x_test, y_train, y_test: Conjuntos de dados de treino e teste para as imagens e rótulos.
    """
    data = pd.read_csv('./fer2013.csv')  # Carrega o dataset contendo as imagens e emoções.
    pixels = data['pixels'].tolist()  # Extrai a coluna 'pixels' contendo as imagens em formato de string.
    # Converte as strings de pixels para arrays NumPy e redimensiona para 48x48 pixels
    faces = np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(48, 48) for pixel in pixels])
    # Converte os rótulos de emoção para uma codificação one-hot
    emotions = to_categorical(data['emotion'], num_classes=7)
    # Normaliza os valores das imagens para o intervalo [0, 1]
    faces = faces.astype('float32') / 255.0
    faces = np.expand_dims(faces, -1)  # Adiciona a dimensão de canal (grayscale)
    # Divide os dados em conjuntos de treino e teste
    return train_test_split(faces, emotions, test_size=0.2, random_state=42)

# Função para construir o modelo de rede neural convolucional (CNN)
def build_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Constrói o modelo de rede neural convolucional (CNN) com camadas de convolução, 
    normalização em lote, pooling e camadas totalmente conectadas.
    
    Args:
        input_shape: Forma de entrada das imagens (altura, largura, canais).
        num_classes: Número de classes de saída (emoções).
        
    Retorna:
        model: Modelo compilado da rede neural.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # Primeira camada convolucional
        BatchNormalization(),  # Normalização em lote
        MaxPooling2D(pool_size=(2, 2)),  # Pooling
        Dropout(0.25),  # Camada de dropout para evitar overfitting
        
        Conv2D(64, (3, 3), activation='relu'),  # Segunda camada convolucional
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),  # Terceira camada convolucional
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        Flatten(),  # Achata a saída das camadas convolucionais para passar para as camadas densas
        Dense(512, activation='relu'),  # Camada densa totalmente conectada
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')  # Camada de saída com ativação softmax para classificação múltipla
    ])
    
    # Compila o modelo com o otimizador Adam, função de perda de entropia cruzada e métrica de acurácia
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Carrega e pré-processa os dados
x_train, x_test, y_train, y_test = load_data()

# Cria o modelo
model = build_model()

# Define os callbacks para o treinamento: EarlyStopping e ModelCheckpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),  # Interrompe o treinamento se a perda de validação não melhorar
    ModelCheckpoint('emotion_model.keras', save_best_only=True, monitor='val_loss')  # Salva o melhor modelo
]

# Treinamento do modelo
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=100, callbacks=callbacks)

# Avaliação do modelo nos dados de teste
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
