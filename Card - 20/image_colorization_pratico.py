import cv2
import numpy as np
import torch
import torch.nn as nn

class ColorizationNet(nn.Module):
    """
    Rede neural para colorização de imagens em tons de cinza.
    - Encoder: Extrai características da imagem em escala de cinza.
    - Decoder: Gera os canais de cor (ab) para reconstruir a imagem em cores.
    """
    def __init__(self):
        """
        Inicializa o modelo com:
        - Encoder: Camada convolucional seguida de ReLU e pooling.
        - Decoder: Camadas convolucionais para gerar os canais de cor.
        """
        super(ColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Extrai características
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduz a resolução
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Expande as características
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)  # Gera canais de cor (ab)
        )

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo.
        - Entrada: Imagem em escala de cinza (1 canal).
        - Saída: Canais de cor (ab) da imagem.
        """
        x = self.encoder(x)  # Codifica as características
        x = self.decoder(x)  # Decodifica para os canais de cor
        return x

# Carrega uma imagem em escala de cinza
gray_image = cv2.imread('gray_image.jpg', 0)  # Lê a imagem como tons de cinza
gray_image = cv2.resize(gray_image, (256, 256))  # Redimensiona para 256x256

# Converte a imagem em tensor normalizado
gray_tensor = torch.tensor(gray_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Adiciona batch e canal

# Instancia o modelo
model = ColorizationNet()

# Passa a imagem pelo modelo para prever os canais de cor
output = model(gray_tensor)  # Saída: canais ab da imagem em cores
