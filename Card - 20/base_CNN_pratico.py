import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Uma CNN simples para classificação de imagens.
    - Conv2d: Extrai características das imagens.
    - MaxPool2d: Reduz a dimensionalidade espacial.
    - Linear: Classifica as imagens em 10 classes.
    """
    def __init__(self):
        """
        Inicializa as camadas:
        - Convolução: 1 canal de entrada, 16 canais de saída, kernel 3x3.
        - Pooling: Reduz tamanho para metade.
        - Linear: Camada totalmente conectada para 10 classes.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Convolução 3x3
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling 2x2
        self.fc1 = nn.Linear(16 * 14 * 14, 10)  # Camada final para 10 classes

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo.
        - Entrada: Imagem 28x28 com 1 canal.
        - Saída: Vetor de 10 classes.
        """
        x = self.pool(torch.relu(self.conv1(x)))  # Convolução + ReLU + Pooling
        x = x.view(-1, 16 * 14 * 14)  # Flatten da saída do pooling
        x = self.fc1(x)  # Camada totalmente conectada
        return x

# Instancia o modelo
model = SimpleCNN()
