import torch
import torch.nn as nn

# Dados de entrada
inputs = torch.tensor([[1.0, -1.0, 2.0], [0.0, 3.0, -3.0]])

# Definição da função de ativação (ReLU)
activation_function = nn.ReLU()

# Aplicando a ativação
activated_outputs = activation_function(inputs)

# Exibindo os resultados
print("Activated Outputs:\n", activated_outputs)
