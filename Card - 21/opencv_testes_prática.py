import cv2
import numpy as np

# Carrega a imagem do diretório especificado.
imagem = cv2.imread('./Card - 21/image.png')

# Converte a imagem de BGR para HSV, um espaço de cores mais adequado para segmentação.
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Define os intervalos de cor para azul.
azul_inferior = np.array([100, 150, 50])
azul_superior = np.array([140, 255, 255])
mascara_azul = cv2.inRange(imagem_hsv, azul_inferior, azul_superior)

# Aplica uma operação de fechamento para melhorar a máscara azul.
kernel_azul = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, kernel_azul)
resultado_azul = cv2.bitwise_and(imagem, imagem, mask=mascara_azul)

# Define os intervalos de cor para vermelho (usando dois intervalos para cobrir o espectro).
vermelho_inferior1 = np.array([0, 150, 50])
vermelho_superior1 = np.array([10, 255, 255])
vermelho_inferior2 = np.array([170, 150, 50])
vermelho_superior2 = np.array([180, 255, 255])
mascara_vermelho1 = cv2.inRange(imagem_hsv, vermelho_inferior1, vermelho_superior1)
mascara_vermelho2 = cv2.inRange(imagem_hsv, vermelho_inferior2, vermelho_superior2)

# Combina as duas máscaras vermelhas.
mascara_vermelho = cv2.addWeighted(mascara_vermelho1, 1.0, mascara_vermelho2, 1.0, 0.0)

# Aplica uma operação de fechamento para melhorar a máscara vermelha.
kernel_vermelho = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mascara_vermelho = cv2.morphologyEx(mascara_vermelho, cv2.MORPH_CLOSE, kernel_vermelho)
resultado_vermelho = cv2.bitwise_and(imagem, imagem, mask=mascara_vermelho)

# Define os intervalos de cor para verde.
verde_inferior = np.array([40, 100, 50])
verde_superior = np.array([80, 255, 255])
mascara_verde = cv2.inRange(imagem_hsv, verde_inferior, verde_superior)

# Aplica uma operação de fechamento para melhorar a máscara verde.
kernel_verde = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel_verde)
resultado_verde = cv2.bitwise_and(imagem, imagem, mask=mascara_verde)

# Exibe os resultados da segmentação.
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Máscara Azul', mascara_azul)
cv2.imshow('Detecção Azul', resultado_azul)
cv2.imshow('Máscara Vermelha', mascara_vermelho)
cv2.imshow('Detecção Vermelha', resultado_vermelho)
cv2.imshow('Máscara Verde', mascara_verde)
cv2.imshow('Detecção Verde', resultado_verde)

# Aguarda o pressionamento de qualquer tecla para encerrar.
cv2.waitKey(0)
cv2.destroyAllWindows()
