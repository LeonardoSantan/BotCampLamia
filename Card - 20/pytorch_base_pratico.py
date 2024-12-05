import torch
import matplotlib.pyplot as plt

tensor_1d = torch.linspace(0, 10, steps=100)
tensor_2d = torch.rand((10, 10))
tensor_3d = torch.rand((10, 10, 3))

plt.figure(figsize=(8, 4))
plt.plot(tensor_1d.numpy(), torch.sin(tensor_1d).numpy(), label="Seno")
plt.plot(tensor_1d.numpy(), torch.cos(tensor_1d).numpy(), label="Cosseno")
plt.title("Visualização de Tensor 1D")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()

plt.figure(figsize=(6, 6))
plt.imshow(tensor_2d.numpy(), cmap='viridis')
plt.title("Visualização de Tensor 2D")
plt.colorbar(label="Valores")
plt.axis("off")

plt.figure(figsize=(6, 6))
plt.imshow(tensor_3d[:, :, 0].numpy(), cmap='plasma')
plt.title("Visualização de Tensor 3D - Camada 1")
plt.colorbar(label="Valores")
plt.axis("off")

plt.show()
