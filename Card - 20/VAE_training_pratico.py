import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def vae_loss(recon_x, x, mu, log_var):
    """
    Calcula a perda do VAE.
    - recon_x: Dados reconstruídos pela rede.
    - x: Dados de entrada originais.
    - mu: Média da distribuição latente.
    - log_var: Logaritmo da variância da distribuição latente.

    A perda combina:
    - reconstruction_loss: Diferença entre os dados reconstruídos e os originais (Binary Cross-Entropy).
    - kl_divergence: Divergência KL para regularização do espaço latente.
    """
    reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')  # BCE para reconstrução
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # Divergência KL
    return reconstruction_loss + kl_divergence


# Transformação dos dados
transform = transforms.Compose([transforms.ToTensor()])  # Converte as imagens para tensores normalizados

# Carregamento do conjunto de treino MNIST
train_data = datasets.MNIST(
    root='./data',  # Diretório para salvar os dados
    train=True,  # Define que é o conjunto de treino
    transform=transform,  # Aplica transformações
    download=True  # Faz o download dos dados se não estiverem disponíveis
)

# DataLoader para gerenciamento de lotes
train_loader = DataLoader(
    train_data,  # Conjunto de dados
    batch_size=32,  # Tamanho do lote
    shuffle=True  # Embaralha os dados a cada época
)

# Instancia o modelo VAE
model = VAE(784, 20)  # VAE com 784 dimensões de entrada e 20 dimensões no espaço latente
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Otimizador Adam com taxa de aprendizado 0.001

# Loop de treinamento
for epoch in range(10):  # Treina por 10 épocas
    for batch_X, _ in train_loader:  # Itera pelos lotes de dados
        batch_X = batch_X.view(-1, 784)  # Achata as imagens 28x28 em vetores de 784 dimensões
        optimizer.zero_grad()  # Zera os gradientes acumulados
        recon_X, mu, log_var = model(batch_X)  # Passa os dados pelo modelo
        loss = vae_loss(recon_X, batch_X, mu, log_var)  # Calcula a perda do VAE
        loss.backward()  # Calcula os gradientes
        optimizer.step()  # Atualiza os pesos do modelo
