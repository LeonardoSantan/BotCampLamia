import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
%matplotlib inline  # Para exibir gráficos diretamente no Jupyter Notebook

# Definindo o conjunto de dados de entrada e saída
x = [[1, 2], [3, 4], [5, 6], [7, 8]]  # Dados de entrada
y = [[3], [7], [11], [15]]  # Dados de saída correspondentes

# Convertendo as listas para tensores PyTorch e ajustando o tipo de dados para float
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

# Definindo o dispositivo para o treinamento (GPU se disponível, caso contrário CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)

# Definindo a arquitetura da rede neural
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()  # Chamando o construtor da classe base nn.Module
        self.layer1 = nn.Linear(2, 8)  # Primeira camada linear com 2 entradas e 8 saídas
        self.activation = nn.ReLU()  # Função de ativação ReLU
        self.layer2 = nn.Linear(8, 1)  # Segunda camada linear com 8 entradas e 1 saída

    def forward(self, x):
        x = self.layer1(x)  # Passando os dados pela primeira camada
        x = self.activation(x)  # Aplicando a função de ativação
        x = self.layer2(x)  # Passando os dados pela segunda camada
        return x

# Instanciando o modelo
model = MyNeuralNet()

# Definindo a função de perda (Erro Quadrático Médio)
loss_func = nn.MSELoss()

# Definindo o otimizador (Gradiente Descendente Estocástico com taxa de aprendizado de 0.001)
opt = SGD(model.parameters(), lr=0.001)

# Lista para armazenar os valores de perda durante o treinamento
losses = []
for _ in range(50):  # Executando o treinamento por 50 épocas
    opt.zero_grad()  # Zerando os gradientes antes de cada época
    loss_value = loss_func(model(X), Y)  # Calculando a perda para o batch atual
    loss_value.backward()  # Calculando os gradientes da perda em relação aos parâmetros
    opt.step()  # Atualizando os parâmetros do modelo com base nos gradientes
    losses.append(loss_value.detach().numpy())  # Armazenando o valor da perda

# Plotando a variação da perda ao longo das épocas
plt.plot(losses)
plt.title('Loss variation over increasing epochs')  # Título do gráfico
plt.xlabel('epochs')  # Rótulo do eixo x
plt.ylabel('loss value')  # Rótulo do eixo y
plt.show()  # Exibindo o gráfico
