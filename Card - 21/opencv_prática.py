import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('./Card - 21/image.png')

# Converter para o espaço de cor HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir os intervalos de cor (exemplo: azul)
lower_bound = np.array([100, 150, 50])  # Matiz, Saturação, Valor mínimos
upper_bound = np.array([140, 255, 255])  # Matiz, Saturação, Valor máximos

# Criar a máscara para a cor desejada
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Refinar a máscara com operações morfológicas
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Aplicar a máscara na imagem original
result = cv2.bitwise_and(image, image, mask=mask)

# Exibir as imagens
cv2.imshow('Imagem Original', image)
cv2.imshow('Máscara', mask)
cv2.imshow('Resultado', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
