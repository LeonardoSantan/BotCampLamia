import torch
import torch.nn as nn
import torch.optim as optim

# Base de treino
inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])  # Entradas: pares de valores
targets = torch.tensor([[3.0], [5.0], [7.0], [9.0]])                    # Saídas correspondentes

# Definição da rede neural
class SimpleNet(nn.Module):
    """
    Rede neural simples com uma única camada totalmente conectada.
    - Entrada: Vetores de tamanho 2.
    - Saída: Um único valor predito.
    """
    def __init__(self):
        """
        Inicializa a camada totalmente conectada:
        - fc: Mapeia dois valores de entrada para uma saída.
        """
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)  # Camada linear com 2 entradas e 1 saída

    def forward(self, x):
        """
        Define a passagem dos dados pela rede.
        - Entrada: Tensor com 2 características.
        - Saída: Valor predito (regressão).
        """
        return self.fc(x)

# Instancia o modelo, a função de perda e o otimizador
model = SimpleNet()
loss_function = nn.MSELoss()  # Erro quadrático médio para medir a perda
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Otimizador SGD com taxa de aprendizado de 0.01

# Loop de treinamento
for epoch in range(100):  # Treina por 100 épocas
    optimizer.zero_grad()  # Zera os gradientes acumulados
    outputs = model(inputs)  # Calcula as predições do modelo
    loss = loss_function(outputs, targets)  # Calcula a perda em relação aos alvos
    loss.backward()  # Calcula os gradientes
    optimizer.step()  # Atualiza os pesos do modelo

# Exibe a perda final após o treinamento
print("Final Loss:", loss.item())
