import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Implementação de um Autoencoder Variacional (VAE).
    - Encoder: Reduz os dados para um espaço latente.
    - Decoder: Reconstrói os dados a partir do espaço latente.
    - Reparameterize: Implementa o truque de reparametrização para aprendizado eficiente.
    """

    def __init__(self, input_dim, latent_dim):
        """
        Inicializa o VAE.
        - input_dim: Dimensão dos dados de entrada.
        - latent_dim: Dimensão do espaço latente.
        """
        super(VAE, self).__init__()
        # Encoder: Reduz os dados para uma representação latente
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),  # Mapeia input_dim para 128 dimensões
            nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)  # Saída para a média (mu)
        self.log_var = nn.Linear(128, latent_dim)  # Saída para o log da variância (log_var)

        # Decoder: Reconstrói os dados a partir do espaço latente
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Mapeia latent_dim para 128 dimensões
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Reconstrói para input_dim
            nn.Sigmoid()  # Garante saída entre 0 e 1
        )

    def reparameterize(self, mu, log_var):
        """
        Aplica o truque de reparametrização para gerar amostras do espaço latente.
        - mu: Média da distribuição latente.
        - log_var: Logaritmo da variância da distribuição latente.
        Retorna amostras latentes calculadas como mu + std * epsilon.
        """
        std = torch.exp(0.5 * log_var)  # Calcula o desvio padrão
        epsilon = torch.randn_like(std)  # Amostra ruído da distribuição normal
        return mu + epsilon * std  # Retorna a amostra reparametrizada

    def forward(self, x):
        """
        Define a passagem dos dados pelo modelo.
        - Entrada: Dados de alta dimensionalidade.
        - Saída: Dados reconstruídos, média (mu), e log_variância (log_var).
        """
        x = self.encoder(x)  # Codifica os dados
        mu = self.mu(x)  # Calcula a média latente
        log_var = self.log_var(x)  # Calcula o log da variância latente
        z = self.reparameterize(mu, log_var)  # Reparametriza para gerar amostras
        return self.decoder(z), mu, log_var  # Decodifica e retorna reconstrução, mu, log_var


# Instancia o modelo VAE com entrada de 784 dimensões (ex.: imagens 28x28) e espaço latente de 20 dimensões
model = VAE(784, 20)

