import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    """
    Um autoencoder convolucional para compressão e reconstrução de imagens.
    - Encoder: Reduz a dimensionalidade das imagens.
    - Decoder: Reconstrói as imagens a partir das características comprimidas.
    """
    def __init__(self):
        """
        Inicializa as camadas do encoder e decoder.
        - Encoder: Duas camadas convolucionais com ReLU e redução de tamanho.
        - Decoder: Duas camadas transpostas convolucionais com ReLU e aumento de tamanho.
        """
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Reduz tamanho e aumenta canais
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # Mais redução e mais canais
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # Aumenta tamanho e reduz canais
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # Reconstrói para 1 canal
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo.
        - Entrada: Imagem 28x28 com 1 canal.
        - Saída: Imagem reconstruída 28x28 com 1 canal.
        """
        x = self.encoder(x)  # Codifica os dados
        x = self.decoder(x)  # Decodifica os dados
        return x

# Instancia o modelo
model = CNNAutoencoder()
