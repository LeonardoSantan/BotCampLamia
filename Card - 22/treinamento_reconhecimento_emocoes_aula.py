#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('rm -rf /kaggle/working/*')


# In[2]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Acessando o dataset

# In[3]:


data = pd.read_csv('./Material/fer2013/fer2013/fer2013.csv')
data.head()


# In[4]:


plt.figure(figsize=(12,6)) # cria uma figura
plt.hist(data['emotion'], bins = 30) # cria um histograma com os valores da coluna emotion
plt.title('images x emotions')


# As classes são:<br>
# 0 - Angry<br>
# 1 - Disgust<br>
# 2 - Fear<br>
# 3 - Happy<br>
# 4 - Sad<br>
# 5 - Surprise<br>
# 6 - Neutral

# # Formatando as imagens

# In[5]:


pixels = data['pixels'].tolist() # series para lista
pixels


# In[6]:


height, width = 48, 48 # altura e largura das imagens orginais, outras dimensões não formarão uma imagem com sentido
faces = [] 
for pixel_sequence in pixels: # itera por cada linha (imagem)
    face = [int(pixel) for pixel in pixel_sequence.split(' ')] # cria uma lista com os pixels formatados para inteiro, através de um split
    face = np.asarray(face).reshape(width, height) # converte pra array e redimensiona
    faces.append(face)


# In[7]:


c = 0
fig, axes = plt.subplots(2,5,figsize=(5,5))
for face in faces[:10]:
    axes.flat[c].imshow(face, cmap='gray')
    axes.flat[c].axis('off')
    c += 1
plt.tight_layout()
plt.show()


# In[8]:


faces = np.array(faces)


# In[9]:


faces.shape


# In[10]:


faces = np.expand_dims(faces, axis = -1) # adiciona uma dimensão unitária (de canal de cor, nesse caso) ao último eixo
faces.shape


# In[11]:


faces = [face.astype('float32')/255. for face in  faces]
faces[0] # visualizando a primeira imagem


# In[12]:


emotions = pd.get_dummies(data['emotion'], dtype=int).values # cria as variáveis dummy (one-hot) e converte para array
emotions


# Códigos One-Hot são úteis para classificações múltiplas, como é o caso.<br>
# Uma alternativa seria com inteiros, utilizando a sparse cross entropy (que utiliza matrizes esparsas) como função de custo.

# # Mais importações

# In[13]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy # problema multiclasse com one-hot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 # importa o L2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model # função de importação de modelos já treinos
from tensorflow.keras.models import model_from_json # função para importação de modelos de arquivo json
from tensorflow.keras import backend as k


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size = 0.1, random_state=42)


# In[15]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)


# In[16]:


print(f'X_train size: {len(X_train)}')
print(f'X_test size: {len(X_test)}')
print(f'X_val size: {len(X_val)}')


# X_val serve para validar a rede após cada epoch, enquanto X_test serve para testar o modelo após o treino completo.

# In[17]:


np.save('mod_xtest', X_test) # salva o conjunto X de teste
np.save('mod_ytest', y_test) # salva o conjunto y de teste


# # Arquitetura da CNN

# In[18]:


num_features = 64 # número de kernels por camada convolucional
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

k.clear_session()
model = Sequential()
model.add(Input((width, height, 1)))
model.add(Conv2D(filters = num_features,
                 kernel_size = (3,3), activation = 'relu', data_format = 'channels_last', # channels_last é o default de data_format
                 kernel_regularizer= l2(0.01)))
model.add(Conv2D(num_features, kernel_size = (3,3), activation='relu',
                 padding = 'same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten()) # planifica a última camada de feature maps, é a camada de entrada da rede densa
model.add(Dense(2*2*2*num_features, activation='relu')) # 1a camada oculta
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu')) #2a camada oculta
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu')) # 3a camada oculta
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation='softmax')) # camada de saída, cada neurônio vai conter a probabilidade para uma emoção

model.summary() # retorna um resumo da estrutura da rede final


# In[19]:


model.compile(optimizer=Adam(learning_rate=0.001, # taxa de aprendizado
                             beta_1=0.9, # termo de decaimento exponencial das estimativas do 1o momento do gradiente
                             beta_2=0.999, # termo de decaimento exponencial das estimativas do 2o momento do gradiente
                             epsilon=1e-7), # termo para não zerar a expressão que multiplica a taxa de aprendizado na atualização dos pesos
                             loss = 'categorical_crossentropy', # problema multiclasse com entradas one-hot
                             metrics=['accuracy']) # já que é um problema de classificação
model_file = 'model_01_expressions.weights.h5' # armazena o conjunto dos pesos
model_file_json = 'model_01_expressions.json' # armazena a estrutura da rede neural

lr_reducer = ReduceLROnPlateau(monitor='val_loss', # verifica o possível plateau pelo val_loss
                               factor = 0.9, # fator de decaimento da lr, que multiplica ela?
                               patience = 3, # espera 3 epochs a partir do momento que a val_loss parou de ter progresso significativo antes de diminuir a lr
                               verbose = 1) # habilita mensagens de aviso no caso de ativação da callback
early_stopper = EarlyStopping(monitor='val_loss', # referência pela qual saber se houve melhoria significativa no aprendizado
                              min_delta = 0, # mudança mínima na quantiadade monitorada para qualificar como uma melhoria significativa
                              patience = 8, # número de epochs seguidas sem melhoria antes de encerrar o treinamento
                              verbose = 1, # mostra uma mensagem indicando a ativação da callback
                              mode = 'auto') # detectar automaticando se melhoria é diminuição ou aumento da métrica monitoriada

checkpointer = ModelCheckpoint(model_file, # arquivo onde serão salvas as versões do modelo
                               monitor = 'val_loss', # vai determinar pela val_loss quando fazer um checkpoint
                               verbose = 1, # vai avisar quando a callback for chamada
                               save_best_only = True, # sobrescreve o modelo anterior com o atual se o atual for melhor
                               save_weights_only = True) 


# In[20]:


model_json = model.to_json() # cria uma versão do modelo convertida para json
with open(model_file_json, 'w') as json_file: # abre o arquivo model_file_json em modo escrita
    json_file.write(model_json) # escreve o json do modelo no arquivo


# In[21]:


history = model.fit(np.array(X_train), np.array(y_train), # dados de treino
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1, # para exibir mensagens de progresso
                    validation_data = (np.array(X_val), np.array(y_val)), # dados de validação
                    shuffle=True, # randomizar os batches
                    callbacks=[lr_reducer, early_stopper, checkpointer]) # define as callbacks


# In[22]:


print(history.history)


# # Visualizando o desempenho do modelo

# In[23]:


def plot_model_history(history):
    fig, axes = plt.subplots(1,2,figsize=(15,5))
    axes[0].plot(range(1, len(history.history['accuracy'])+1), # coord x, epoch 
                 history.history['accuracy'], # coord y, accuracy
                 'r', # cor da curva
                 label='training accuracy') 
    axes[0].plot(range(1, len(history.history['val_accuracy'])+1), # coord x, epoch
                 history.history['val_accuracy'], # coord y, validation accuracy
                 'b', # cor da curva
                 label='validation accuracy') 
    axes[0].set_title('accuracy x epoch')
    axes[0].set_ylabel('accuracy')
    axes[0].set_ylabel('epoch')
    axes[0].legend() # define legendas para as respectivas curvas

    axes[1].plot(range(1, len(history.history['loss'])+1), # coord x, epoch 
                 history.history['loss'], # coord y, loss
                 'r', # cor da curva
                 label='training loss') 
    axes[1].plot(range(1, len(history.history['val_loss'])+1), # coord x, epoch
                 history.history['val_loss'], # coord y, validation loss
                 'b', # cor da curva
                 label='validation loss') 
    axes[1].set_title('loss x epoch')
    axes[1].set_ylabel('loss')
    axes[1].set_ylabel('epoch')
    axes[1].legend() # define legendas para as respectivas curvas
    fig.savefig('model_history_model01.png')
plot_model_history(history)


# # Testando o modelo com a base de teste

# In[24]:


scores = model.evaluate(np.array(X_test), np.array(y_test), 
                        batch_size=batch_size)


# In[25]:


scores


# Erro e precisão, respectivamente.

# In[26]:


print(f'Accuracy: {scores[1]}\nLoss: {scores[0]}')


# # Geração da matriz de confunsão

# In[27]:


true_y = []
pred_y = []
x = np.load('mod_xtest.npy') # carrega os dados previsores de teste
y = np.load('mod_ytest.npy') # carrega os dados de classe de teste


# In[28]:


x[0]


# In[29]:


y[0]


# In[30]:


json_file = open(model_file_json, 'r') # abre o arquivo json da estrutura do modelo
loaded_model_json = json_file.read() # lê o conteúdo do arquivo json
json_file.close() # fecha o arquivo


# In[31]:


loaded_model = model_from_json(loaded_model_json) # cria um modelo tensoflow a partir do json
loaded_model.load_weights(model_file) # adiciona os pesos do arquivo model_01_expressions.weights.h5


# In[32]:


y_pred = loaded_model.predict(x)
y_pred


# In[33]:


yp = y_pred.tolist() # lista com as previsões
yt = y.tolist() # lista com as previsões
len(yp)


# In[34]:


pred_y = [] # lista que conterá as classes previstas para cada amostra
true_y = [] # lista que conterá as classes verdadeiras de cada amostra
count = 0 # variável que irá armazenar o número de previsões corretas
for i in range(len(yp)):
    yy = max(yp[i]) # retorna o maior valor de yp[i]
    yyp = max(yt[i]) # retorna o maior valor de yt[i]
    pred_y.append(yp[i].index(yy)) # add a pred_y o índice de yp em que yy está, que corresponde à classe
    true_y.append(yt[i].index(yyp)) # add a true_y o índice de yt em que yyt está
    if (yp[i].index(yy) == yt[i].index(yyp)): # se a previsão é igual ao valor verdadeira
        count += 1 # aumenta em 1 o número de previsões corretas
acc = (count / len(y)) * 100
print(f'Model accuracy for the test set: {acc:.2f}%.')


# In[35]:


np.save('truey_mod01', true_y)
np.save('predy_mod01', pred_y)


# In[36]:


from sklearn.metrics import confusion_matrix # classe para geração de matriz de confusão

y_pred = np.load('predy_mod01.npy')
y_true = np.load('truey_mod01.npy')


# In[37]:


cm = confusion_matrix(y_true, y_pred) # gera a matriz de confusão
emotions = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']
title = 'Confusion Matrix'
cm


# Quase nenhuma imagem foi classificada como nojo, principalmente porque o conjunto de faces com nojo nos dados de treino era bem baixo.

# In[38]:


import itertools
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # cria uma imagem, interpolando os pixels com o valor mais próximo, numa escala de azul, com os valores da matriz de confusão
plt.title(title)
plt.colorbar() # adiciona uma barra de cor indicando a escala
tick_marks = np.arange(len(emotions)) # cria valores para posicionar marcas de eixo
plt.xticks(tick_marks, emotions, rotation = 45); # adiciona as labels de emoção nos pontos definidos por tick_marks no eixo x
plt.yticks(tick_marks, emotions); # adiciona as labels ao eixo y
fmt = 'd' # variável de formatação
thresh = cm.max() / 2. # valor que define o limite para mudança de cor da fonte
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # itera por todas as combinações de linha e coluna possíveis
    plt.text(j, i, # posição da imagem a se adicionar o texto
             format(cm[i, j], fmt), # o texto em si
             horizontalalignment='center', # alinha o texto ao centro do retângulo
             color = 'white' if cm[i,j] > thresh else 'black') # define a cor da fonte de acordo com o tom de azul do retângulo
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.savefig('confusion_matrix_mod01.png') # salva a matriz de confusão em png


# # Testando o modelo novamente

# In[41]:


image = cv2.imread('./Material/testes/teste02.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converte a imagem para rgb
plt.imshow(image)


# In[42]:


original = image.copy()
gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) # converte para cinza
plt.imshow(gray, cmap='gray')


# In[43]:


face_cascade = cv2.CascadeClassifier('./Material/haarcascade_frontalface_default.xml') # carrega o modelo para extração da ROI
faces = face_cascade.detectMultiScale(gray, 1.1, 3) # extrai as ROIs
faces


# In[44]:


for (x, y, w, h) in faces:
    cv2.rectangle(original, (x, y), (x+w, y+h), (255, 0, 0), 1) # adiciona um retângulo em volta da face
    roi_gray = gray[y: y+h, x: x+w] # extrai a ROI da imagem em questão
    roi_gray = roi_gray.astype('float') / 255. # converte para float e normaliza
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) # adiciona dimensão de canal de cor e de batch
    prediction = loaded_model.predict(cropped_img)[0] # faz a previsão para a imagem em questão
    cv2.putText(original, emotions[int(np.argmax(prediction))], (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
plt.imshow(original)


# In[ ]:




import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Carregar o dataset FER2013 para análise de emoções faciais
print("Carregando o dataset FER2013...")
data = pd.read_csv('./Material/fer2013/fer2013/fer2013.csv')

# Verificar a estrutura inicial dos dados
print("Exibindo as primeiras linhas do dataset:")
print(data.head())

# Exibir a distribuição das classes para entender a proporção entre emoções
plt.figure(figsize=(12, 6))
plt.hist(data['emotion'], bins=30, color='skyblue', alpha=0.7)
plt.title('Distribuição de Emoções no Dataset')
plt.xlabel('Emoção')
plt.ylabel('Frequência')
plt.show()

# Pré-processar as imagens: Converter pixels de string para arrays numéricos
print("Processando imagens do dataset...")
pixels = data['pixels'].tolist()
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    faces.append(face)

faces = np.expand_dims(np.array(faces), axis=-1)
faces = faces.astype('float32') / 255.0  # Normalizar os valores dos pixels

# Criar as representações one-hot para as emoções
emotions = pd.get_dummies(data['emotion'], dtype=int).values

# Dividir os dados em conjuntos de treino, validação e teste
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

# Exibir o tamanho dos conjuntos gerados
print(f"Conjunto de treino: {len(X_train)} amostras")
print(f"Conjunto de validação: {len(X_val)} amostras")
print(f"Conjunto de teste: {len(X_test)} amostras")

# Salvar os conjuntos de teste para análises posteriores
np.save('mod_xtest.npy', X_test)
np.save('mod_ytest.npy', y_test)

# Configurar a arquitetura do modelo CNN
print("Configurando a arquitetura do modelo CNN...")
num_features = 64
num_labels = 7
width, height = 48, 48

model = Sequential([
    Input((width, height, 1)),
    Conv2D(num_features, (3, 3), activation='relu', kernel_regularizer=l2(0.01), padding='same'),
    Conv2D(num_features, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Conv2D(2 * num_features, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(2 * num_features, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_labels, activation='softmax')
])

# Exibir a estrutura do modelo configurado
model.summary()

# Compilar o modelo com as configurações de otimizador e função de perda
print("Compilando o modelo para treinamento...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Configurar callbacks para ajuste dinâmico do treinamento
model_file = 'model_emotions.h5'
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, verbose=1),
    ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=1)
]

# Treinar o modelo com os dados de treino e validação
print("Iniciando o treinamento do modelo...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# Avaliar o desempenho do modelo no conjunto de teste
print("Avaliando o modelo com o conjunto de teste...")
scores = model.evaluate(X_test, y_test)
print(f"Perda no teste: {scores[0]:.4f}, Acurácia no teste: {scores[1]:.4f}")

# Visualizar os gráficos de desempenho durante o treinamento
def plot_model_history(history):
    print("Gerando gráficos de desempenho...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history['accuracy'], label='Acurácia - Treino', color='green')
    axes[0].plot(history.history['val_accuracy'], label='Acurácia - Validação', color='orange')
    axes[0].set_title('Acurácia por Época')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Acurácia')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Perda - Treino', color='green')
    axes[1].plot(history.history['val_loss'], label='Perda - Validação', color='orange')
    axes[1].set_title('Perda por Época')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Perda')
    axes[1].legend()

    plt.show()

plot_model_history(history)
