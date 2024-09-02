# Importando as bibliotecas necessárias
from torchvision import datasets  # Para baixar e carregar datasets
import torch  # Biblioteca principal para operações com tensores e treinamento de modelos
import matplotlib.pyplot as plt  # Para visualização de dados
import numpy as np  # Biblioteca para operações com arrays

# Definindo o diretório onde o dataset será salvo
data_folder = '~/data/FMNIST'

# Baixando o dataset Fashion MNIST e carregando os dados de treino
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)

# Extraindo os dados e rótulos das imagens
tr_images = fmnist.data
tr_targets = fmnist.targets

# Configurando o matplotlib para exibir gráficos no Jupyter Notebook
%matplotlib inline

# Exibindo amostras das imagens de treino
R, C = len(tr_targets.unique()), 10  # Número de linhas baseado no número de classes únicas e 10 colunas
fig, ax = plt.subplots(R, C, figsize=(10,10))  # Criando uma grade de subplots
for label_class, plot_row in enumerate(ax):
    # Encontrando índices das imagens pertencentes à classe atual
    label_x_rows = np.where(tr_targets == label_class)[0]
    for plot_cell in plot_row:
        plot_cell.grid(False)  # Desativando a grade
        plot_cell.axis('off')  # Desativando os eixos
        ix = np.random.choice(label_x_rows)  # Selecionando um índice aleatório
        x, y = tr_images[ix], tr_targets[ix]  # Obtendo a imagem e o rótulo
        plot_cell.imshow(x, cmap='gray')  # Exibindo a imagem em escala de cinza
plt.tight_layout()  # Ajustando o layout para evitar sobreposição

# Definindo a classe de dataset personalizada
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn  # Para definir redes neurais
device = "cuda" if torch.cuda.is_available() else "cpu"  # Usando GPU se disponível

class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float()  # Convertendo os dados para float
        x = x.view(-1, 28*28)  # Achatar cada imagem para um vetor de 784 pixels
        self.x, self.y = x, y  # Armazenando os dados e rótulos

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]  # Obtendo o item do índice fornecido
        return x.to(device), y.to(device)  # Movendo os dados para o dispositivo correto (GPU/CPU)

    def __len__(self):
        return len(self.x)  # Retornando o número total de exemplos

# Função para obter o DataLoader
def get_data():
    train = FMNISTDataset(tr_images, tr_targets)  # Criando o dataset
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)  # Criando o DataLoader com tamanho de batch 32
    return trn_dl

# Definindo o modelo, função de perda e otimizador
from torch.optim import SGD
def get_model():
    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),  # Camada linear com 1000 neurônios
        nn.ReLU(),  # Função de ativação ReLU
        nn.Linear(1000, 10)  # Camada de saída com 10 neurônios (uma para cada classe)
    ).to(device)  # Movendo o modelo para o dispositivo correto (GPU/CPU)
    loss_fn = nn.CrossEntropyLoss()  # Função de perda para classificação
    optimizer = SGD(model.parameters(), lr=1e-2)  # Otimizador SGD com taxa de aprendizado de 0.01
    return model, loss_fn, optimizer

# Função para calcular a acurácia
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()  # Colocando o modelo em modo de avaliação
    prediction = model(x)  # Obtendo as previsões do modelo
    max_values, argmaxes = prediction.max(-1)  # Obtendo os índices das previsões com maior valor
    is_correct = argmaxes == y  # Comparando as previsões com os rótulos verdadeiros
    return is_correct.cpu().numpy().tolist()  # Retornando a lista de acurácia como numpy array

# Função para treinar um batch
def train_batch(x, y, model, opt, loss_fn):
    model.train()  # Colocando o modelo em modo de treinamento
    prediction = model(x)  # Obtendo as previsões do modelo
    batch_loss = loss_fn(prediction, y)  # Calculando a perda do batch
    batch_loss.backward()  # Calculando os gradientes
    opt.step()  # Atualizando os parâmetros do modelo
    opt.zero_grad()  # Limpando os gradientes para o próximo batch
    return batch_loss.item()  # Retornando a perda como valor escalar

# Criando DataLoader, modelo, função de perda e otimizador
trn_dl = get_data()
model, loss_fn, optimizer = get_model()

# Treinando o modelo
losses, accuracies = [], []
for epoch in range(5):  # Treinando por 5 épocas
    print(epoch)
    epoch_losses, epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch  # Obtendo um batch de dados
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)  # Treinando o batch
        epoch_losses.append(batch_loss)  # Adicionando a perda do batch à lista
    epoch_loss = np.array(epoch_losses).mean()  # Calculando a perda média da época
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch  # Obtendo um batch de dados
        is_correct = accuracy(x, y, model)  # Calculando a acurácia do batch
        epoch_accuracies.extend(is_correct)  # Adicionando a acurácia à lista
    epoch_accuracy = np.mean(epoch_accuracies)  # Calculando a acurácia média da época
    losses.append(epoch_loss)  # Adicionando a perda média da época à lista
    accuracies.append(epoch_accuracy)  # Adicionando a acurácia média da época à lista

# Plotando as perdas e acurácias
epochs = np.arange(5) + 1
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])  # Convertendo ticks para porcentagem
plt.legend()

# Atualizando a classe FMNISTDataset para normalizar os dados
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / 255.  # Normalizando os dados para o intervalo [0, 1]
        x = x.view(-1, 28*28)  # Achatar cada imagem para um vetor de 784 pixels
        self.x, self.y = x, y  # Armazenando os dados e rótulos

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]  # Obtendo o item do índice fornecido
        return x.to(device), y.to(device)  # Movendo os dados para o dispositivo correto (GPU/CPU)

    def __len__(self):
        return len(self.x)  # Retornando o número total de exemplos
