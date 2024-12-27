from torchvision import transforms

# Define uma sequência de transformações para pré-processar imagens
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),      # Aplica flip horizontal aleatório
    transforms.RandomRotation(10),          # Rotaciona a imagem aleatoriamente até 10 graus
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Recorta e redimensiona aleatoriamente para 28x28 pixels
    transforms.ToTensor()                   # Converte a imagem para um tensor
])
