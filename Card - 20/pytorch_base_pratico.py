import torch
import matplotlib.pyplot as plt

# Criação de tensores
tensor_1d = torch.linspace(0, 10, steps=100)  # Tensor 1D com 100 valores espaçados entre 0 e 10
tensor_2d = torch.rand((10, 10))              # Tensor 2D (10x10) com valores aleatórios
tensor_3d = torch.rand((10, 10, 3))           # Tensor 3D (10x10x3) com valores aleatórios

# Visualização do tensor 1D: Funções seno e cosseno
plt.figure(figsize=(8, 4))
plt.plot(tensor_1d.numpy(), torch.sin(tensor_1d).numpy(), label="Seno")     # Seno do tensor 1D
plt.plot(tensor_1d.numpy(), torch.cos(tensor_1d).numpy(), label="Cosseno")  # Cosseno do tensor 1D
plt.title("Visualização de Tensor 1D")   # Título do gráfico
plt.xlabel("X")                          # Rótulo do eixo X
plt.ylabel("Y")                          # Rótulo do eixo Y
plt.legend()                             # Legenda para identificar as curvas
plt.grid()                               # Grade no gráfico

# Visualização do tensor 2D: Mapa de cores
plt.figure(figsize=(6, 6))
plt.imshow(tensor_2d.numpy(), cmap='viridis')  # Exibição do tensor 2D com mapa de cores 'viridis'
plt.title("Visualização de Tensor 2D")        # Título do gráfico
plt.colorbar(label="Valores")                 # Barra de cores com rótulo
plt.axis("off")                               # Remove os eixos

# Visualização do tensor 3D: Uma camada (dimensão 0) do tensor
plt.figure(figsize=(6, 6))
plt.imshow(tensor_3d[:, :, 0].numpy(), cmap='plasma')  # Exibição da camada 1 do tensor 3D
plt.title("Visualização de Tensor 3D - Camada 1")      # Título do gráfico
plt.colorbar(label="Valores")                          # Barra de cores com rótulo
plt.axis("off")                                        # Remove os eixos

# Exibe todos os gráficos
plt.show()
