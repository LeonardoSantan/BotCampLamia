import torch
import torch.nn as nn
import torch.optim as optim

# Base de treino
inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
targets = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# Rede neural
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()

print("Final Loss:", loss.item())
