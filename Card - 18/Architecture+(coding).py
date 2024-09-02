import torch
from torchviz import make_dot

# Definição dos dados de entrada X e dos valores alvo Y
# X representa pares de números, e Y é a soma dos elementos desses pares.
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]

# Conversão das listas para tensores para serem usados com PyTorch
# O método .float() converte o tipo de dado para ponto flutuante (float).
X = torch.tensor(x).float()
Y = torch.tensor(y).float()

# Determina se o código será executado em GPU (cuda) ou CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)  # Exibe se está utilizando 'cuda' (GPU) ou 'cpu'

# Move os tensores X e Y para o dispositivo especificado (GPU ou CPU)
X = X.to(device)
Y = Y.to(device)

# Importação do módulo de redes neurais do PyTorch
import torch.nn as nn

# Definição de uma rede neural personalizada
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

# Geração de um gráfico da arquitetura da rede neural e salvamento como PNG
make_dot(model(X), params=dict(model.named_parameters())).render("dense_network", format="png")

# Exibe o formato do tensor X
X.shape  # Saída: torch.Size([4, 2])


# Exibe os pesos da primeira camada (layer1) do modelo
model.layer1.weight


# Exibe o primeiro conjunto de parâmetros do modelo, que corresponde aos pesos da primeira camada
model.parameters().__next__()
