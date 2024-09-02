# Importando bibliotecas necessárias
import cv2 as cv  # OpenCV para manipulação de imagens
import matplotlib.pyplot as plt  # Matplotlib para exibição de imagens
import numpy as np  # NumPy para manipulação de arrays

# Configurando o Jupyter Notebook para exibir gráficos inline
%matplotlib inline

# Lendo a imagem do arquivo
img = cv.imread('ab6761610000e5eb31f6ab67e6025de876475814')

# Exibindo a imagem original
plt.imshow(img)
plt.title('Imagem Original')  # Adiciona título para a imagem
plt.show()

# Convertendo a imagem de BGR (formato padrão do OpenCV) para RGB
rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# Exibindo a imagem no formato RGB
plt.imshow(rgb_image)
plt.title('Imagem RGB')  # Adiciona título para a imagem RGB
plt.show()

# Convertendo a imagem RGB para escala de cinza
gray_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2GRAY)
# Exibindo a imagem em escala de cinza
plt.imshow(gray_image, cmap='gray')
plt.title('Imagem em Escala de Cinza')  # Adiciona título para a imagem em escala de cinza
plt.show()

# Imprimindo as dimensões da imagem RGB
print("Dimensões da imagem RGB:", rgb_image.shape)
# Imprimindo as dimensões da imagem em escala de cinza
print("Dimensões da imagem em escala de cinza:", gray_image.shape)

# Redimensionando a imagem em escala de cinza para 50x50 pixels
img_gray_small = cv.resize(gray_image, (50, 50))
# Exibindo a imagem redimensionada
plt.imshow(img_gray_small, cmap='gray')
plt.title('Imagem em Escala de Cinza (50x50)')  # Adiciona título para a imagem redimensionada
plt.show()

# Exibindo uma parte específica da imagem em escala de cinza (da linha 10 até 340, e todas as colunas)
plt.imshow(gray_image[10:340, :], cmap='gray')
plt.title('Parte da Imagem em Escala de Cinza')  # Adiciona título para a parte da imagem
plt.show()

# Exibindo uma parte específica da imagem em escala de cinza (todas as linhas, e até a coluna 300)
plt.imshow(gray_image[:, :300], cmap='gray')
plt.title('Outra Parte da Imagem em Escala de Cinza')  # Adiciona título para a outra parte da imagem
plt.show()
