import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Carregamento do conjunto de dados MNIST
train_data = MNIST(
    root='./data',             # Diretório para salvar os dados
    train=True,                # Define que é o conjunto de treino
    download=True,             # Faz o download se os dados não estiverem disponíveis
    transform=transform        # Aplica transformações às imagens (não definido aqui)
)

# DataLoader para gerenciamento dos lotes
train_loader = DataLoader(
    train_data,                # Conjunto de dados
    batch_size=32,             # Tamanho do lote
    shuffle=True               # Embaralha os dados em cada época
)

# Define a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()            # Perda para classificação
optimizer = optim.Adam(model.parameters(),   # Adam com parâmetros do modelo
                       lr=0.001)             # Taxa de aprendizado

# Loop de treinamento
for epoch in range(5):  # Treina por 5 épocas
    for images, labels in train_loader:  # Itera pelos lotes de dados
        optimizer.zero_grad()           # Zera os gradientes acumulados
        outputs = model(images)         # Faz a previsão com o modelo
        loss = criterion(outputs, labels)  # Calcula a perda
        loss.backward()                 # Computa o gradiente
        optimizer.step()                # Atualiza os parâmetros do modelo
