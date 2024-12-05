import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Carregar imagem
image = cv2.imread('imagem.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Redimensionar e normalizar
resized_image = cv2.resize(gray_image, (128, 128)).astype(np.float32)
scaler = MinMaxScaler()
normalized_image = scaler.fit_transform(resized_image.flatten().reshape(-1, 1)).reshape(128, 128)

# Criar dataset simulado (imagens transformadas em vetor)
X = np.stack([normalized_image for _ in range(100)], axis=0)
y = np.random.randint(0, 2, size=(100,))  # Classes binárias
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Converter para tensores
X_train = torch.tensor(X_train).unsqueeze(1)  # Adicionando dimensão de canal
X_val = torch.tensor(X_val).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Modelo Deep Learning
class DeepNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(DeepNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (input_shape // 4) ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Hiperparâmetros
input_shape = 128
num_classes = 2
learning_rate = 0.01
batch_size = 16
epochs = 10

# Tunning de hiperparâmetros
def train_model(learning_rate, batch_size):
    model = DeepNet(input_shape=input_shape, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

best_model = train_model(learning_rate, batch_size)
