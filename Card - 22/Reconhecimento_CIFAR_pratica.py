import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Função para carregar e pré-processar os dados do CIFAR-10
def load_data():
    """
    Carrega o dataset CIFAR-10, realiza o pré-processamento das imagens e rótulos.
    As imagens são normalizadas para o intervalo [0, 1] e os rótulos são convertidos para one-hot encoding.

    Retorna:
        x_train, x_test, y_train, y_test: Conjuntos de dados de treino e teste para as imagens e rótulos.
    """
    # Carrega os dados do CIFAR-10 diretamente do Keras
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normaliza as imagens para o intervalo [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Converte os rótulos para one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, x_test, y_train, y_test


# Função para construir o modelo de rede neural convolucional (CNN)
def build_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Constrói o modelo de rede neural convolucional (CNN) para classificação de imagens CIFAR-10.

    Args:
        input_shape: Forma de entrada das imagens (altura, largura, canais).
        num_classes: Número de classes de saída.

    Retorna:
        model: Modelo compilado da rede neural.
    """
    model = Sequential([
        # Primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Terceira camada convolucional
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Camadas densas
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compila o modelo com o otimizador Adam, função de perda de entropia cruzada e métrica de acurácia
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_history(history):
    """
    Plota os gráficos de perda e acurácia durante o treinamento e validação.

    Args:
        history: Histórico de treinamento retornado pela função model.fit().
    """
    # Plot da perda
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    # Plot da acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.show()


def main():
    # Carrega e pré-processa os dados
    x_train, x_test, y_train, y_test = load_data()

    # Divide o conjunto de treinamento em treinamento e validação
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Cria o modelo
    model = build_model()

    # Define os callbacks para o treinamento: EarlyStopping e ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('cifar10_model.h5', save_best_only=True, monitor='val_loss')
    ]

    # Treinamento do modelo
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=64,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )

    # Avaliação do modelo nos dados de teste
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f'Perda no Teste: {test_loss:.4f}')
    print(f'Acurácia no Teste: {test_accuracy:.4f}')

    # Plot dos gráficos de treinamento
    plot_history(history)


if __name__ == "__main__":
    main()
