import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def vae_loss(recon_x, x, mu, log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = VAE(784, 20)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_X, _ in train_loader:
        batch_X = batch_X.view(-1, 784)  # Flatten the image
        optimizer.zero_grad()
        recon_X, mu, log_var = model(batch_X)
        loss = vae_loss(recon_X, batch_X, mu, log_var)
        loss.backward()
        optimizer.step()
