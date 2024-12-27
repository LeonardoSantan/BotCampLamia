import torch.nn as nn

class DeepFakeModel(nn.Module):
    """
    Um modelo de autoencoder convolucional para reconstrução de imagens,
    projetado para tarefas como geração ou modificação de imagens.
    - Encoder: Reduz a dimensionalidade e extrai características principais.
    - Decoder: Reconstrói a imagem original a partir das características comprimidas.
    """
    def __init__(self):
        """
        Inicializa o modelo com:
        - Encoder: Duas camadas convolucionais para compressão das imagens.
        - Decoder: Duas camadas convolucionais transpostas para reconstrução das imagens.
        """
        super(DeepFakeModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # Reduz tamanho e aumenta canais
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Mais redução e mais canais
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # Aumenta tamanho e reduz canais
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # Reconstrói para 3 canais (RGB)
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo.
        - Entrada: Imagem RGB com 3 canais.
        - Saída: Imagem reconstruída RGB com 3 canais.
        """
        x = self.encoder(x)  # Codifica a imagem
        x = self.decoder(x)  # Decodifica a imagem
        return x

# Instancia o modelo
model = DeepFakeModel()
