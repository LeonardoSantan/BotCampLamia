import numpy as np
# Transfer function

def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0

def signoidFunction(soma):
    return 1 / (1+ np.exp(-soma))

def tahmFunction(soma):
    return(np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0

def sofmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

valores = [5.0, 2.0, 1.3]

print(sofmaxFunction(valores))
