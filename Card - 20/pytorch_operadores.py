import torch
import matplotlib.pyplot as plt

tensor_1d = torch.linspace(0, 10, steps=100)
tensor_2d = torch.rand((10, 10))
tensor_3d = torch.rand((10, 10, 3))

tensor_1d_unsqueezed = tensor_1d.unsqueeze(1)
tensor_ones = torch.ones_like(tensor_2d)
tensor_sum = tensor_2d + tensor_ones
tensor_product = torch.matmul(tensor_2d, tensor_ones.T)
tensor_3d_mean = tensor_3d.mean(dim=2)

plt.figure(figsize=(8, 4))
plt.plot(tensor_1d.numpy(), tensor_1d_unsqueezed.squeeze().numpy(), label="Tensor com Dimensão Extra")
plt.title("Tensor 1D com Unsqueeze")
plt.xlabel("X")
plt.ylabel("Valor")
plt.legend()
plt.grid()

plt.figure(figsize=(6, 6))
plt.imshow(tensor_sum.numpy(), cmap='viridis')
plt.title("Soma de Tensor 2D com Tensor de Uns")
plt.colorbar(label="Valores")
plt.axis("off")

plt.figure(figsize=(6, 6))
plt.imshow(tensor_product.numpy(), cmap='plasma')
plt.title("Produto Matricial")
plt.colorbar(label="Valores")
plt.axis("off")

plt.figure(figsize=(6, 6))
plt.imshow(tensor_3d_mean.numpy(), cmap='gray')
plt.title("Tensor 3D Média (Escala de Cinza)")
plt.colorbar(label="Valores")
plt.axis("off")

plt.show()
