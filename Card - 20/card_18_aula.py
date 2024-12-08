import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# === Tensores e Operações Básicas === #
# Definição de tensores e operações comuns
x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print("Shape:", x.shape)  # Retorna as dimensões do tensor
print("Multiplicação escalar:", x * 10)  # Multiplica cada elemento por 10
print("Adição escalar:", x.add(10))  # Soma 10 a cada elemento
print("Mudança de shape (4x2):", x.view(4, 2))  # Altera a forma do tensor

# Redimensionamento de tensores
a = torch.ones(2, 1, 10)
print("Shape original:", a.shape)
print("Shape após squeeze:", a.squeeze(1).shape)  # Remove dimensões de tamanho 1

# Multiplicação de matrizes
y = torch.tensor([[1, 2, 3], [2, 3, 4], [4, 5, 6], [7, 8, 9]])
print("Multiplicação matricial:", torch.matmul(x, y))  # Produto de matrizes

# === Rede Neural Básica (MLP) === #
# Rede neural simples com uma camada oculta
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)  # Camada totalmente conectada
        self.activation = nn.ReLU()  # Função de ativação
        self.layer2 = nn.Linear(8, 1)  # Saída com 1 neurônio

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)

# Dados de entrada e saída para treinamento
data_x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.float32)
data_y = torch.tensor([[3], [7], [11], [15]], dtype=torch.float32)

# Treinamento do modelo
model = MyNeuralNet()
loss_func = nn.MSELoss()  # Função de perda (Erro quadrático médio)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Otimizador SGD

for epoch in range(50):
    optimizer.zero_grad()  # Zera os gradientes
    loss = loss_func(model(data_x), data_y)  # Calcula a perda
    loss.backward()  # Calcula os gradientes
    optimizer.step()  # Atualiza os pesos

print("Treinamento da rede neural básica concluído.")

# === CNN com FashionMNIST === #
# Preprocessamento de imagens
transform = transforms.Compose([transforms.ToTensor()])
fmnist = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
dataloader = DataLoader(fmnist, batch_size=32, shuffle=True)

# Definição do modelo CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3)  # Convolução
        self.pool = nn.MaxPool2d(2)  # Pooling
        self.fc = nn.Linear(64 * 13 * 13, 10)  # Camada totalmente conectada

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 64 * 13 * 13)
        return self.fc(x)

# Treinamento do modelo CNN
cnn_model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

for epoch in range(5):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

print("Treinamento do modelo CNN concluído.")

# Visualização de um exemplo
test_images, test_labels = next(iter(dataloader))
predicted = cnn_model(test_images).argmax(dim=1)
plt.imshow(test_images[0][0], cmap='gray')
plt.title(f'Label: {test_labels[0]} | Predicted: {predicted[0]}')
plt.show()

# === Autoencoder Vanilla === #
# Definição de autoencoder básico
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, latent_dim))  # Codificação
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(True),
            nn.Linear(64, 128), nn.ReLU(True),
            nn.Linear(128, 28 * 28), nn.Tanh())  # Decodificação

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(len(x), 1, 28, 28)

# Treinamento do autoencoder
autoencoder = AutoEncoder(3).to('cpu')
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(5):
    for images, _ in dataloader:
        optimizer.zero_grad()
        output = autoencoder(images)
        loss = criterion(output, images)
        loss.backward()
        optimizer.step()

print("Treinamento do autoencoder concluído.")

# === Variational Autoencoder === #
# Definição de um VAE (autoencoder variacional)
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU())  # Codificação
        self.fc_mu = nn.Linear(64 * 7 * 7, 20)  # Média
        self.fc_logvar = nn.Linear(64 * 7 * 7, 20)  # Log variância
        self.decoder = nn.Sequential(
            nn.Linear(20, 64 * 7 * 7),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid())  # Decodificação

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.decoder(z.view(z.size(0), 64, 7, 7))
        return z, mu, logvar

# Treinamento do VAE
vae = VAE().to('cpu')
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(5):
    for images, _ in dataloader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(images)
        recon_loss = nn.functional.mse_loss(recon, images)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_div
        loss.backward()
        optimizer.step()

print("Treinamento do VAE concluído.")
