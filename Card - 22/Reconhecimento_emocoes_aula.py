import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Exibindo a versão do TensorFlow instalada
print(f"Versão do TensorFlow: {tf.__version__}")

# -------------------------
# Configurações iniciais
# -------------------------

# Carregando a imagem para análise
def carregar_imagem(caminho):
    """
    Carrega uma imagem a partir de um caminho e converte para RGB.
    """
    img = cv2.imread(caminho)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

# Caminho da imagem de teste
teste_img_path = './Material/testes/teste_gabriel.png'
img = carregar_imagem(teste_img_path)

# -------------------------
# Carregando os modelos
# -------------------------

# Modelo para detecção de faces
cascade_faces = 'Material/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cascade_faces)

# Modelo para detecção de emoções
model_path = 'Material/modelo_01_expressoes.h5'
classifier_emotion = load_model(model_path, compile=False)

# Classes de emoções reconhecidas pelo modelo
emotions = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']

# -------------------------
# Detecção de faces na imagem
# -------------------------

def detectar_faces(img):
    """
    Detecta faces em uma imagem usando o modelo Haarcascade.
    Retorna as coordenadas das faces detectadas.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    return faces

faces = detectar_faces(img)
print(f"Faces detectadas: {faces}")

# -------------------------
# Análise da emoção
# -------------------------

def analisar_emocao(img, face_coords):
    """
    Realiza a análise de emoção em uma face detectada.
    """
    x, y, w, h = face_coords
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    roi = gray[y:y+h, x:x+w]

    # Redimensionando para o tamanho esperado pelo modelo
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype('float') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # Predição da emoção
    preds = classifier_emotion.predict(roi)[0]
    emotion_prob = np.max(preds)
    emotion_label = emotions[preds.argmax()]

    print(f"Emoção detectada: {emotion_label} com {emotion_prob * 100:.2f}% de certeza.")

    return emotion_label, emotion_prob

# Processando a primeira face detectada (se houver)
if len(faces) > 0:
    label, prob = analisar_emocao(img, faces[0])

    # Adicionando o texto e retângulo na imagem original
    x, y, w, h = faces[0]
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    plt.imshow(img)
    plt.axis('off')
    plt.show()
else:
    print("Nenhuma face detectada na imagem.")

# -------------------------
# Processamento em lote de múltiplas imagens
# -------------------------

def processar_multiplas_imagens(folder_path):
    """
    Detecta faces e emoções em todas as imagens de uma pasta.
    """
    imgs = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = carregar_imagem(img_path)

        faces = detectar_faces(img)
        for face_coords in faces:
            label, prob = analisar_emocao(img, face_coords)

            x, y, w, h = face_coords
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        imgs.append(img)

    # Exibindo a última imagem processada
    plt.imshow(imgs[-1])
    plt.axis('off')
    plt.show()

imgs_folder = './Material/testes'
processar_multiplas_imagens(imgs_folder)
