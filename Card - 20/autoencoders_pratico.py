import torch
import torch.nn as nn

class VanillaAutoencoder(nn.Module):
    """
    Um autoencoder simples com encoder e decoder.
    - Encoder: Reduz os dados de 784 para 64 dimensões.
    - Decoder: Reconstrói os dados de 64 para 784 dimensões.
    """
    def __init__(self):
        """
        Define as camadas do encoder e decoder.
        """
        super(VanillaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),  # Reduz para 128 dimensões
            nn.ReLU(),
            nn.Linear(128, 64),  # Reduz para 64 dimensões
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),  # Aumenta para 128 dimensões
            nn.ReLU(),
            nn.Linear(128, 784),  # Reconstrói para 784 dimensões
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo.
        - Entrada: Dados de 784 dimensões.
        - Saída: Dados reconstruídos de 784 dimensões.
        """
        x = self.encoder(x)  # Codifica os dados
        x = self.decoder(x)  # Decodifica os dados
        return x

# Instancia o modelo
model = VanillaAutoencoder()
