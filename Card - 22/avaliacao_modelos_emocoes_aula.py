import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import operator

# Carregar os modelos para avaliação
model_files = ['modelo_01_expressoes.h5', 'modelo_02_expressoes.h5', 
               'modelo_03_expressoes.h5', 'modelo_04_expressoes.h5',
               'modelo_05_expressoes.h5']
models = {}

# Carregar os conjuntos de teste
x_test = np.load('./Material/mod_xtest.npy')
y_test = np.load('./Material/mod_ytest.npy')

# Avaliar cada modelo e armazenar os resultados
for model_file in model_files:
    model_path = f'./Material/{model_file}'
    loaded_model = load_model(model_path)
    scores = loaded_model.evaluate(x_test, y_test, batch_size=64, verbose=0)
    print(f"Modelo: {model_file} | Loss: {scores[0]:.4f} | Accuracy: {scores[1]:.4f}")
    models[model_file] = scores[1]

# Ordenar os modelos por acurácia\ordered_models = sorted(models.items(), key=operator.itemgetter(1), reverse=True)
print("\nModelos ordenados por acurácia:")
for model, acc in ordered_models:
    print(f"{model}: {acc:.4f}")

# Selecionar o melhor modelo
best_model_path = f'./Material/{ordered_models[0][0]}'
print(f"\nMelhor modelo: {ordered_models[0][0]} com acurácia {ordered_models[0][1]:.4f}")

# Testar o melhor modelo em uma imagem
image_path = './Material/testes/teste_gabriel.png'
image = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Configuração para detecção de faces e classificação
haarcascade_faces = './Material/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_faces)
emotion_classifier = load_model(best_model_path, compile=False)
emotions = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']

# Processar a imagem para detecção de emoções
original = image.copy()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

if len(faces) > 0:
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_prob = np.max(preds)
        emotion_label = emotions[np.argmax(preds)]

        # Adicionar a emoção detectada à imagem
        cv2.putText(original, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("Nenhuma face detectada.")

# Exibir a imagem processada
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
