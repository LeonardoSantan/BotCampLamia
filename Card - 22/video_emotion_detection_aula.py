import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Carregar o modelo treinado para reconhecimento de emoções
model = load_model('/kaggle/input/arquivos-card-20/Material/modelo_02_expressoes.h5')

# Configuração do arquivo de vídeo para análise
video_file = '/kaggle/input/arquivos-card-20/Material/Videos/video_teste04.mp4'
cap = cv2.VideoCapture(video_file)
conected, video = cap.read()
if not conected:
    raise ValueError("Erro ao carregar o vídeo. Certifique-se de que o caminho está correto.")
print(f"Dimensões do vídeo: {video.shape}")

# Redimensionar o vídeo para processamento mais rápido
resize = True
max_width = 600
if resize and video.shape[1] > max_width:
    proportion = video.shape[1] / video.shape[0]
    video_width = max_width
    video_height = int(video_width / proportion)
else:
    video_width = video.shape[1]
    video_height = video.shape[0]

# Configuração do arquivo de saída
file_name = 'resultado_video_teste04.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
video_output = cv2.VideoWriter(file_name, fourcc, fps, (video_width, video_height))

# Configurações para detecção e reconhecimento de emoções
haarcascade_faces = '/kaggle/input/arquivos-card-20/Material/haarcascade_frontalface_default.xml'
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
font = cv2.FONT_HERSHEY_SIMPLEX
small_font, big_font = 0.4, 0.7

# Processar cada frame do vídeo
while True:
    conected, frame = cap.read()
    if not conected:
        print("Processamento finalizado.")
        break

    start_time = time.time()

    if resize:
        frame = cv2.resize(frame, (video_width, video_height))

    # Converter o frame para escala de cinza para detecção de faces
    face_cascade = cv2.CascadeClassifier(haarcascade_faces)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame atual
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Processar cada face detectada
    for (x, y, w, h) in faces:
        # Desenhar um retângulo ao redor da face detectada
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 50, 50), 2)

        # Extrair a região de interesse (ROI) e preparar para o modelo
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Realizar a predição da emoção
        preds = model.predict(roi)[0]
        emotion_label = np.argmax(preds)

        # Adicionar o texto com a emoção detectada no frame
        cv2.putText(frame, emotions[emotion_label], (x, y - 10), font, big_font, (255, 255, 255), 1, cv2.LINE_AA)

    # Adicionar o tempo de processamento do frame no vídeo
    processing_time = time.time() - start_time
    cv2.putText(
        frame,
        f'Frame processed in {processing_time:.2f} seconds',
        (20, video_height - 20),
        font,
        small_font,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Adicionar o frame processado ao vídeo de saída
    video_output.write(frame)

# Liberar os recursos após o processamento
video_output.release()
cap.release()
print("O vídeo processado foi salvo com sucesso.")
