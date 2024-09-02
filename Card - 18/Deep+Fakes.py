import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_snippets import *
from random_warp import get_training_data

# Cria o diretório se ele não existir e baixa os arquivos necessários
if not os.path.exists('Faceswap-Deepfake-Pytorch'):
    !wget -q https://www.dropbox.com/s/5ji7jl7httso9ny/person_images.zip
    !wget -q https://raw.githubusercontent.com/sizhky/deep-fake-util/main/random_warp.py
    !unzip -q person_images.zip

# Instala bibliotecas necessárias
!pip install -q torch_snippets torch_summary

# Carrega o classificador de rosto pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(img):
    """
    Detecta e recorta o rosto da imagem, redimensionando para 256x256 pixels.
    Args:
        img (ndarray): Imagem para detecção de rosto.
    Returns:
        tuple: (imagem recortada, booleano indicando se o rosto foi detectado)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img2 = img[y:(y + h), x:(x + w), :]
        img2 = cv2.resize(img2, (256, 256))
        return img2, True
    else:
        return img, False

# Cria diretórios para armazenar as imagens recortadas
!mkdir cropped_faces_personA
!mkdir cropped_faces_personB

def crop_images(folder):
    """
    Aplica a função de recorte de rosto em todas as imagens do diretório especificado.
    Args:
        folder (str): Diretório contendo as imagens.
    """
    images = Glob(folder + '/*.jpg')
    for i in range(len(images)):
        img = read(images[i], 1)
        img2, face_detected = crop_face(img)
        if not face_detected:
            continue
        else:
            cv2.imwrite('cropped_faces_' + folder + '/' + str(i) + '.jpg', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

# Processa imagens das duas pastas
crop_images('personA')
crop_images('personB')

class ImageDataset(Dataset):
    """
    Dataset personalizado para imagens de rosto, que ajusta as imagens para a mesma média.
    """
    def __init__(self, items_A, items_B):
        self.items_A = np.concatenate([read(f, 1)[None] for f in items_A]) / 255.
        self.items_B = np.concatenate([read(f, 1)[None] for f in items_B]) / 255.
        self.items_A += self.items_B.mean(axis=(0, 1, 2)) - self.items_A.mean(axis=(0, 1, 2))

    def __len__(self):
        return min(len(self.items_A), len(self.items_B))

    def __getitem__(self, ix):
        a, b = choose(self.items_A), choose(self.items_B)
        return a, b

    def collate_fn(self, batch):
        """
        Agrupa o batch e prepara os dados para treinamento.
        Args:
            batch (list): Lista de tuplas contendo pares de imagens.
        Returns:
            tuple: (imagens A, imagens B, targets A, targets B) como tensores
        """
        imsA, imsB = list(zip(*batch))
        imsA, targetA = get_training_data(imsA, len(imsA))
        imsB, targetB = get_training_data(imsB, len(imsB))
        imsA, imsB, targetA, targetB = [torch.Tensor(i).permute(0, 3, 1, 2).to(device) for i in [imsA, imsB, targetA, targetB]]
        return imsA, imsB, targetA, targetB

# Cria dataset e dataloader para as imagens processadas
a = ImageDataset(Glob('cropped_faces_personA'), Glob('cropped_faces_personB'))
x = DataLoader(a, batch_size=32, collate_fn=a.collate_fn)

inspect(*next(iter(x)))

# Exibe imagens do dataloader
for i in next(iter(x)):
    subplots(i[:8], nc=4, sz=(4, 2))

def _ConvLayer(input_features, output_features):
    """
    Camada convolucional com LeakyReLU.
    Args:
        input_features (int): Número de canais de entrada.
        output_features (int): Número de canais de saída.
    Returns:
        nn.Sequential: Sequência de camadas.
    """
    return nn.Sequential(
        nn.Conv2d(input_features, output_features, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(0.1, inplace=True)
    )

def _UpScale(input_features, output_features):
    """
    Camada de upscale usando convolução transposta com LeakyReLU.
    Args:
        input_features (int): Número de canais de entrada.
        output_features (int): Número de canais de saída.
    Returns:
        nn.Sequential: Sequência de camadas.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(input_features, output_features, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(0.1, inplace=True)
    )

class Reshape(nn.Module):
    """
    Camada para redimensionar o tensor.
    """
    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)  # Redimensiona para (batch_size, 1024, 4, 4)
        return output

class Autoencoder(nn.Module):
    """
    Autoencoder para transformação de imagens usando duas decodificações distintas.
    """
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            _ConvLayer(3, 128),
            _ConvLayer(128, 256),
            _ConvLayer(256, 512),
            _ConvLayer(512, 1024),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 1024),
            nn.Linear(1024, 1024 * 4 * 4),
            Reshape(),
            _UpScale(1024, 512),
        )

        # Decodificador para imagem A
        self.decoder_A = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Decodificador para imagem B
        self.decoder_B = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, select='A'):
        """
        Faz a passagem de imagem pelo encoder e decodificador selecionado.
        Args:
            x (Tensor): Imagem de entrada.
            select (str): Decodificador a ser usado ('A' ou 'B').
        Returns:
            Tensor: Imagem reconstruída.
        """
        if select == 'A':
            out = self.encoder(x)
            out = self.decoder_A(out)
        else:
            out = self.encoder(x)
            out = self.decoder_B(out)
        return out

def train_batch(model, data, criterion, optimizers):
    """
    Treina o modelo em um batch.
    Args:
        model (nn.Module): Modelo a ser treinado.
        data (tuple): Dados de entrada (imagens e targets).
        criterion (nn.Module): Função de perda.
        optimizers (tuple): Otimizadores para decodificadores A e B.
    Returns:
        tuple: Perda para imagens A e B.
    """
    optA, optB = optimizers
    optA.zero_grad()
    optB.zero_grad()
    imgA, imgB, targetA, targetB = data
    _imgA, _imgB = model(imgA, 'A'), model(imgB, 'B')

    lossA = criterion(_imgA, targetA)
    lossB = criterion(_imgB, targetB)

    lossA.backward()
    lossB.backward()

    optA.step()
    optB.step()

    return lossA.item(), lossB.item()

# Inicializa o modelo e move para o dispositivo
model = Autoencoder().to(device)

# Cria dataset e dataloader
dataset = ImageDataset(Glob('cropped_faces_personA'), Glob('cropped_faces_personB'))
dataloader = DataLoader(dataset, 32, collate_fn=dataset.collate_fn)

# Inicializa otimizadores e função de perda
optimizers = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model
