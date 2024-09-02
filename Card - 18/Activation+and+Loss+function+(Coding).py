import math

# Função de ativação sigmoid (1/(1 + e^(-x)))
# Essa função é comumente usada em redes neurais para introduzir não-linearidade.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

import torch
from torch import nn

# Implementação de uma camada ReLU personalizada
# ReLU (Rectified Linear Unit) é uma função de ativação amplamente utilizada em redes neurais.
class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # A função retorna o máximo entre zero e o valor de entrada x.
        # Isso efetivamente "corta" valores negativos, convertendo-os em zero.
        return torch.max(torch.zeros_like(x), x)

import numpy as np

# Função para calcular o erro quadrático médio (MSE)
# Essa função é frequentemente usada como função de perda para problemas de regressão.
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

import numpy as np

# Função para calcular a perda de entropia cruzada
# A entropia cruzada é uma função de perda comum para problemas de classificação.
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Valor mínimo para y_pred para evitar divisão por zero
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Limita os valores de y_pred
    # Calcula a entropia cruzada para duas distribuições y_true (verdadeiros) e y_pred (preditos)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
