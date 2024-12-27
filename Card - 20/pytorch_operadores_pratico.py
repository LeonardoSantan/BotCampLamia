import torch
import matplotlib.pyplot as plt

# Criação de tensores
tensor_1d = torch.linspace(0, 10, steps=100)  # Tensor 1D com 100 valores espaçados uniformemente entre 0 e 10
tensor_2d = torch.rand((10, 10))              # Tensor 2D (10x10) com valores aleatórios
tensor_3d = torch.rand((10, 10, 3))           # Tensor 3D (10x10x3) com valores aleatórios

# Operações nos tensores
tensor_1d_unsqueezed = tensor_1d.unsqueeze(1)  # Adiciona uma dimensão extra ao tensor 1D
tensor_ones = torch.ones_like(tensor_2d)       # Cria um tensor 2D de uns com as mesmas dimensões que tensor_2d
tensor_sum = tensor_2d + tensor_ones           # Soma elemento a elemento entre tensor_2d e tensor_ones
tensor_product = torch.matmul(tensor_2d, tensor_ones.T)  # Produto matricial entre tensor_2d e a transposta de tensor_ones
tensor_3d_mean = tensor_3d.mean(dim=2)         # Calcula a média ao longo da dimensão 2 do tensor 3D

# Visualização do tensor 1D com dimensão extra
plt.figure(figsize=(8, 4))
plt.plot(tensor_1d.numpy(), tensor_1d_unsqueezed.squeeze().numpy(), label="Tensor com Dimensão Extra")  # Plot do tensor 1D
plt.title("Tensor 1D com Unsqueeze")   # Título do gráfico
plt.xlabel("X")                        # Rótulo do eixo X
plt.ylabel("Valor")                    # Rótulo do eixo Y
plt.legend()                           # Adiciona legenda
plt.grid()                             # Adiciona grade ao gráfico

# Visualização da soma do tensor 2D com tensor de uns
plt.figure(figsize=(6, 6))
plt.imshow(tensor_sum.numpy(), cmap='viridis')  # Exibe o tensor somado com o mapa de cores 'viridis'
plt.title("Soma de Tensor 2D com Tensor de Uns")  # Título do gráfico
plt.colorbar(label="Valores")  # Adiciona barra de cores
plt.axis("off")                # Remove os eixos

# Visualização do produto matricial
plt.figure(figsize=(6, 6))
plt.imshow(tensor_product.numpy(), cmap='plasma')  # Exibe o produto matricial com o mapa de cores 'plasma'
plt.title("Produto Matricial")   # Título do gráfico
plt.colorbar(label="Valores")    # Adiciona barra de cores
plt.axis("off")                  # Remove os eixos

# Visualização da média ao longo da dimensão 2 do tensor 3D
plt.figure(figsize=(6, 6))
plt.imshow(tensor_3d_mean.numpy(), cmap='gray')  # Exibe a média com escala de cinza
plt.title("Tensor 3D Média (Escala de Cinza)")  # Título do gráfico
plt.colorbar(label="Valores")  # Adiciona barra de cores
plt.axis("off")                # Remove os eixos

# Exibe os gráficos
plt.show()
