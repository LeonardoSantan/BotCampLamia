import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

def load_image(image_path, transform):
    """
    Carrega e transforma uma imagem.
    - image_path: Caminho para a imagem.
    - transform: Transformações a serem aplicadas na imagem.
    Retorna a imagem transformada como um tensor com dimensão de batch.
    """
    image = Image.open(image_path).convert('RGB')  # Abre a imagem em RGB
    image = transform(image).unsqueeze(0)         # Aplica transformações e adiciona dimensão do batch
    return image

# Define as transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensiona a imagem para 256x256
    transforms.ToTensor()           # Converte a imagem para tensor
])

# Carrega a imagem de conteúdo e de estilo
content_image = load_image('content.jpg', transform)  # Imagem de conteúdo
style_image = load_image('style.jpg', transform)      # Imagem de estilo
generated_image = content_image.clone().requires_grad_(True)  # Copia a imagem de conteúdo para iniciar o gerado

# Modelo pré-treinado VGG-19 para extração de características
model = models.vgg19(pretrained=True).features.eval()  # Usa apenas as camadas de features

# Otimizador para ajustar a imagem gerada
optimizer = optim.Adam([generated_image], lr=0.01)  # Ajusta apenas os pixels da imagem gerada

# Loop de otimização para transferência de estilo
for step in range(300):  # Itera por 300 passos
    optimizer.zero_grad()  # Zera os gradientes acumulados
    generated_features = model(generated_image)  # Extrai características da imagem gerada
    content_features = model(content_image)      # Extrai características da imagem de conteúdo
    style_features = model(style_image)          # Extrai características da imagem de estilo

    # Cálculo da perda (exemplo simplificado)
    loss = (torch.mean((generated_features - content_features) ** 2) +  # Perda de conteúdo
            torch.mean((generated_features - style_features) ** 2))    # Perda de estilo

    loss.backward()  # Calcula os gradientes
    optimizer.step()  # Atualiza os pixels da imagem gerada
