import cv2

# Carregar a imagem
image = cv2.imread('./Card - 20/image.png')

# Converter para escala de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Redimensionar a imagem
resized_image = cv2.resize(gray_image, (100, 100))

# Aplicar filtro Gaussiano
blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

# Detecção de bordas com Canny
edges = cv2.Canny(blurred_image, 50, 150)

# Exibir imagens
cv2.imshow('Original', image)
cv2.imshow('Cinza', gray_image)
cv2.imshow('Redimensionada', resized_image)
cv2.imshow('Bordas', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
