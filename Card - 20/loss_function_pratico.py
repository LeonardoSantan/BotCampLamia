import torch
import torch.nn as nn

# Dados de entrada e saída
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
targets = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

# Definição de um modelo simples
model = nn.Linear(2, 2)

# Função de perda
loss_function = nn.MSELoss()

# Saída do modelo
outputs = model(inputs)

# Cálculo da perda
loss = loss_function(outputs, targets)

# Exibindo a perda
print("Loss:", loss.item())
