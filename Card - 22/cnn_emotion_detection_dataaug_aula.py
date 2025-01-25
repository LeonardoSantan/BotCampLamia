import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Carregando o dataset que contém imagens e emoções representadas por classes

# Caminho do arquivo
data = pd.read_csv('/kaggle/input/arquivos-card-20/Material/fer2013/fer2013/fer2013.csv')

# Exibindo as primeiras linhas para verificação
print(data.head())

# Visualizando a distribuição das classes no conjunto de dados
plt.figure(figsize=(12, 6))
plt.hist(data['emotion'], bins=30)
plt.title('Distribuição de Imagens por Emoção')
plt.show()

# Verificando desbalanceamento nas classes usando z-score
value_counts = data['emotion'].value_counts()
zscores = zscore(value_counts)
plt.bar(value_counts.index, zscores)
plt.title('Z-Score da Distribuição das Classes')
plt.show()

# Pré-processamento das imagens
pixels = data['pixels'].to_list()
width, height = 48, 48
faces = []

# Visualização de algumas imagens convertidas
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
i = 0
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    faces.append(face)

    if i < 10:
        axes.flat[i].imshow(cv2.cvtColor(face.astype(np.uint8), cv2.COLOR_GRAY2RGB))
        axes.flat[i].axis('off')
    i += 1

plt.tight_layout()
plt.show()

# Normalizando os valores das imagens para o intervalo [0, 1]
def normalizar(x):
    x = x.astype('float32') / 255.0
    return x

faces = np.expand_dims(np.asarray(faces), -1)
faces = normalizar(faces)

# Convertendo as emoções em codificação one-hot
dummies = pd.get_dummies(data['emotion'], dtype=np.uint8)
emotions = dummies.to_numpy()
print(f"Total de imagens no dataset: {len(faces)}")

# Divisão dos dados em treino, validação e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

print(f'Conjunto de treino: {len(X_train)} imagens')
print(f'Conjunto de validação: {len(X_val)} imagens')
print(f'Conjunto de teste: {len(X_test)} imagens')

# Salvando os conjuntos de teste para análises futuras
np.save('mod_xtest', X_test)
np.save('mod_ytest', y_test)

# Arquitetura da Rede Neural Convolucional (CNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

num_features = 32
num_classes = 7
batch_size = 64
epochs = 100

model = Sequential()

# Adicionando camadas convolucionais e de pooling
model.add(Conv2D(filters=num_features, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', input_shape=(width, height, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(num_features, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# Repetindo os blocos de convolução com aumento de filtros
model.add(Conv2D(filters=2*num_features, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2*num_features, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

# Data Augmentation para aumentar a variedade de dados de treino
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
steps_per_epoch = X_train.shape[0] // batch_size

# Configurando os callbacks para o treinamento
model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

model_file = 'modelo_02_expressoes_dataaug.weights.h5'
model_file_json = 'modelo_02_expressoes_dataaug.json'
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
checkpointer = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

# Salvando a arquitetura do modelo
model_json = model.to_json()
with open(model_file_json, 'w') as json_file:
    json_file.write(model_json)

# Treinamento do modelo com os dados aumentados
history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, validation_data=(X_val, y_val), callbacks=[lr_reducer, early_stopper, checkpointer])

# Função para plotar o histórico do modelo
def plot_model_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history.history['accuracy'], label='Treinamento')
    axs[0].plot(history.history['val_accuracy'], label='Validação')
    axs[0].set_title('Acurácia por Época')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Treinamento')
    axs[1].plot(history.history['val_loss'], label='Validação')
    axs[1].set_title('Perda por Época')
    axs[1].legend()
    plt.show()

plot_model_history(history)
