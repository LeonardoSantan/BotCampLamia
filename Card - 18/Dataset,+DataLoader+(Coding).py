from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

# Definição dos dados de entrada X e dos valores alvo Y
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]

# Conversão das listas para tensores para serem usados com PyTorch
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

# Determina se o código será executado em GPU (cuda) ou CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)

# Definição de um dataset personalizado usando a classe Dataset do PyTorch
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float().to(device)  # Conversão e movimentação para o dispositivo
        self.y = torch.tensor(y).float().to(device)

    def __len__(self):
        return len(self.x)  # Retorna o número de amostras no dataset

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]  # Retorna uma amostra do dataset pelo índice

# Instanciação do dataset
ds = MyDataset(x, y)

# Criação de um DataLoader para carregar os dados em batches
dl = DataLoader(ds, batch_size=2, shuffle=True)  # Batch size define o número de amostras por lote

# Iteração sobre os dados usando o DataLoader
for x, y in dl:
    print(x, y)  # Exibe os batches de dados carregados

import torch
import torch.nn as nn
from torch.optim import SGD

# Definição da classe da rede neural personalizada
class MyNeuralNet(nn.Module):
    def __init__(self):
        # Chama o construtor da classe pai para garantir a herança correta
        super().__init__()
        # Definição das camadas da rede neural
        self.layer1 = nn.Linear(2, 8)  # Camada linear que mapeia 2 entradas para 8 saídas
        self.activation = nn.ReLU()  # Função de ativação ReLU (Rectified Linear Unit)
        self.layer2 = nn.Linear(8, 1)  # Camada linear que mapeia 8 entradas para 1 saída

    # Método que define o fluxo de dados pela rede neural
    # É chamado automaticamente quando passamos dados pelo modelo
    def forward(self, x):
        x = self.layer1(x)  # Passa o dado pela primeira camada linear
        x = self.activation(x)  # Aplica a função de ativação ReLU
        x = self.layer2(x)  # Passa o dado pela segunda camada linear
        return x  # Retorna o resultado final

# Instanciação do modelo de rede neural definido acima
model = MyNeuralNet()

# Definição da função de perda MSE (Mean Squared Error)
loss_func = nn.MSELoss()

# Definição do otimizador SGD (Stochastic Gradient Descent)
opt = SGD(model.parameters(), lr=0.001)  # lr define a taxa de aprendizado

# Lista para armazenar os valores das perdas durante o treinamento
losses = []

# Treinamento do modelo por 50 épocas
for _ in range(50):  # Executa por 50 épocas
    for data in dl:  # Itera sobre os dados no DataLoader
        opt.zero_grad()  # Zera os gradientes antes de cada passo
        x1, y1 = data
        loss_value = loss_func(model(x1), y1)  # Calcula a perda entre a predição e o valor real
        loss_value.backward()  # Calcula os gradientes em relação aos parâmetros do modelo

        # Atualiza os pesos do modelo com base nos gradientes e no algoritmo de otimização
        opt.step()

        # Armazena o valor da perda atual na lista
        losses.append(loss_value.detach().numpy())
