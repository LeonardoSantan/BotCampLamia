import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications import VGG16

# Carregando o arquivo CSV contendo informações sobre as imagens e emoções
# Este arquivo possui a representação em pixels e a classe emocional correspondente para cada imagem
data_folder = './Material/fer2013/fer2013/fer2013.csv'
data = pd.read_csv(data_folder)
print(data.head())

# Visualizando a distribuição das classes emocionais no conjunto de dados
# Isso ajuda a entender se há desbalanceamento entre as classes
plt.figure(figsize=(12, 6))
plt.hist(data['emotion'], bins=30)
plt.title('Distribuição de Imagens por Emoção')
plt.show()

# Pré-processando os dados para preparar as imagens para o treinamento do modelo
pixels = data['pixels'].tolist()
height, width = 48, 48

# Convertendo a representação em pixels para matrizes de imagens
# Além disso, as imagens são convertidas para RGB pois a VGG16 espera este formato
fig, axs = plt.subplots(2, 5, figsize=(5, 5))
faces = []
samples = 0
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height, 1)
    face = np.asarray(np.dstack((face, face, face)), dtype=np.uint8)
    faces.append(face)

    if samples < 10:
        axs.flat[samples].imshow(face)
        axs.flat[samples].axis('off')
    samples += 1

# Normalizando os valores das imagens para que estejam entre 0 e 1
# Isso ajuda a melhorar a estabilidade e desempenho do modelo durante o treinamento
faces = np.asarray(faces).astype('float32') / 255.

# Convertendo as emoções para o formato one-hot encoding
# Cada emoção será representada como um vetor binário, onde apenas uma posição é 1
emotions = pd.get_dummies(data['emotion']).to_numpy()

# Dividindo o conjunto de dados em treino, teste e validação
# A divisão garante que o modelo seja avaliado em dados que ele não viu durante o treinamento
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

# Exibindo a quantidade de imagens em cada conjunto
print(f'Treinamento: {len(X_train)}, Teste: {len(X_test)}, Validação: {len(X_val)}')

# Salvando os conjuntos de teste para análises futuras
np.save('mod_xtest', X_test)
np.save('mod_ytest', y_test)

# Configurando o modelo usando Transfer Learning com a VGG16
# A VGG16 é uma rede pré-treinada na base ImageNet e será ajustada para o conjunto atual
num_classes = 7
batch_size = 16
epochs = 30

# Carregando a VGG16 sem a parte densa original, pois ela será adaptada para o problema atual
vgg = VGG16(input_shape=(width, height, 3), weights='imagenet', include_top=False)
vgg.trainable = False

# Adicionando camadas específicas para a classificação de emoções
# GlobalAveragePooling reduz a dimensionalidade, e a camada Dense final realiza a predição
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(num_classes, activation='softmax')

# Construindo o modelo final
model = Sequential([
    vgg,
    global_average_layer,
    prediction_layer
])
model.summary()

# Compilando o modelo com a função de perda categórica e o otimizador Adam
# O uso de métricas como acurácia ajuda a monitorar o desempenho do modelo
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

# Configurando callbacks para controle do treinamento
# Inclui redução de taxa de aprendizado, parada antecipada e salvamento do melhor modelo
arquivo_modelo = "modelo_vgg_expressoes.weights.h5"
arquivo_modelo_json = "modelo_vgg_expressoes.json"
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
checkpointer = ModelCheckpoint(arquivo_modelo, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

# Salvando a arquitetura do modelo em um arquivo JSON
model_json = model.to_json()
with open(arquivo_modelo_json, "w") as json_file:
    json_file.write(model_json)

# Iniciando o treinamento do modelo com os dados de treino e validação
history = model.fit(np.array(X_train), np.array(y_train),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(np.array(X_val), np.array(y_val)),
                    shuffle=True,
                    callbacks=[lr_reducer, early_stopper, checkpointer])

# Função para exibir gráficos de desempenho do modelo durante o treinamento
def plot_model_history(history):
    """
    Gera gráficos para visualizar a acurácia e perda ao longo das épocas.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de Acurácia
    axs[0].plot(history.history['accuracy'], 'r', label='Treinamento')
    axs[0].plot(history.history['val_accuracy'], 'b', label='Validação')
    axs[0].set_title('Acurácia por Época')
    axs[0].set_xlabel('Época')
    axs[0].set_ylabel('Acurácia')
    axs[0].legend()

    # Gráfico de Perda
    axs[1].plot(history.history['loss'], 'r', label='Treinamento')
    axs[1].plot(history.history['val_loss'], 'b', label='Validação')
    axs[1].set_title('Perda por Época')
    axs[1].set_xlabel('Época')
    axs[1].set_ylabel('Perda')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Exibindo os gráficos do histórico de treinamento
plot_model_history(history)

