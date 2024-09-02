from torchvision import datasets
import torch

# Definindo o diretório onde o dataset será salvo
data_folder = '/home/oem/Github/BotCampLamia/Card - 18/'

# Carregando o dataset FashionMNIST para treinamento e validação
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data  # Imagens de treinamento
tr_targets = fmnist.targets  # Labels de treinamento

val_fmnist = datasets.FashionMNIST(data_folder, download=True, train=False)
val_images = val_fmnist.data  # Imagens de validação
val_targets = val_fmnist.targets  # Labels de validação

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# Verifica se a GPU está disponível e define o dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Definição da classe customizada para o dataset
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / 255  # Normaliza as imagens dividindo por 255
        x = x.view(-1, 1, 28, 28)  # Redimensiona as imagens para 1x28x28
        self.x, self.y = x, y

    def __getitem__(self, ix):
        # Retorna o tensor da imagem e o label correspondente
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    def __len__(self):
        # Retorna o número total de amostras no dataset
        return len(self.x)

from torch.optim import SGD, Adam

# Definição do modelo CNN usando nn.Sequential
def get_model():
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),  # Primeira camada convolucional
        nn.MaxPool2d(kernel_size=2),  # Camada de pooling
        nn.ReLU(),  # Função de ativação ReLU
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),  # Segunda camada convolucional
        nn.MaxPool2d(kernel_size=2),  # Camada de pooling
        nn.ReLU(),  # Função de ativação ReLU
        nn.Flatten(),  # Achata o tensor para uma camada linear
        nn.Linear(in_features=3200, out_features=256),  # Primeira camada linear
        nn.ReLU(),  # Função de ativação ReLU
        nn.Linear(in_features=256, out_features=10)  # Camada linear final para classificação em 10 classes
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()  # Função de perda de entropia cruzada
    optimizer = Adam(model.parameters(), lr=1e-3)  # Otimizador Adam
    return model, loss_fn, optimizer

# Função para treinar o modelo em um batch
def train_batch(x, y, model, opt, loss_fn):
    prediction = model(x)  # Faz a previsão com o modelo
    batch_loss = loss_fn(prediction, y)  # Calcula a perda
    batch_loss.backward()  # Propaga o erro para calcular gradientes
    optimizer.step()  # Atualiza os pesos do modelo
    optimizer.zero_grad()  # Zera os gradientes para o próximo batch
    return batch_loss.item()

# Função para calcular a acurácia
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()  # Coloca o modelo em modo de avaliação
    prediction = model(x)  # Faz a previsão
    max_values, argmaxes = prediction.max(-1)  # Pega a classe com maior valor
    is_correct = argmaxes == y  # Compara a previsão com o label real
    return is_correct.cpu().numpy().tolist()  # Retorna uma lista indicando acertos

# Função para carregar os dados de treinamento e validação
def get_data():
    train_dataset = FMNISTDataset(tr_images, tr_targets)  # Dataset de treinamento
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # DataLoader para treinamento

    val_dataset = FMNISTDataset(val_images, val_targets)  # Dataset de validação
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_images), shuffle=True)  # DataLoader para validação

    return train_dataloader, val_dataloader

# Função para calcular a perda no conjunto de validação
@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()  # Coloca o modelo em modo de avaliação
    prediction = model(x)  # Faz a previsão
    val_loss = loss_fn(prediction, y)  # Calcula a perda
    return val_loss.item()

# Carrega os dados e o modelo
trn_dl, val_dl = get_data()
model, loss_fn, optimizer = get_model()

# Listas para armazenar perdas e acurácias
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# Loop de treinamento por epochs
for epoch in range(5):
    print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []

    # Treinamento
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    # Acurácia de treinamento
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    # Validação
    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        validation_loss = val_loss(x, y, model, loss_fn)
    val_epoch_accuracy = np.mean(val_is_correct)

    # Armazena os resultados
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(validation_loss)
    val_accuracies.append(val_epoch_accuracy)

# Plot das perdas e acurácias
epochs = np.arange(5) + 1
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
%matplotlib inline

# Plot de perdas de treinamento e validação
plt.subplot(211)
plt.plot(epochs, train_losses, 'bo', label='Training loss')
plt.plot(epochs, val_losses, 'r', label='Validation loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation loss with CNN')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

# Plot de acurácias de treinamento e validação
plt.subplot(212)
plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy with CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()

# Gera previsões para uma imagem específica deslocada horizontalmente
preds = []
ix = 2210
for px in range(-5, 6):
    img = tr_images[ix] / 255.  # Normaliza a imagem
    img = img.view(28, 28)  # Redimensiona para 28x28
    img2 = np.roll(img, px, axis=1)  # Desloca a imagem horizontalmente
    img3 = torch.Tensor(img2).view(-1, 1, 28, 28).to(device)  # Converte para tensor e move para o dispositivo
    np_output = model(img3).cpu().detach().numpy()  # Faz a previsão e move para a CPU
    pred = np.exp(np_output) / np.sum(np.exp(np_output))  # Calcula a softmax para gerar probabilidades
    preds.append(pred)  # Armazena a previsão
    plt.imshow(img2)  # Mostra a imagem deslocada
    plt.title(fmnist.classes[pred[0].argmax()])  # Exibe a classe prevista
    plt.show()
