import torch
import plotly.graph_objs as go
import plotly.offline as pyo

# Criação de um tensor 1D com dois elementos
sample = torch.tensor([10, 11])
print(sample.shape)  # Exibe a forma do tensor: torch.Size([2])

# Criação de um tensor 2D (matriz) de 2x2
x = torch.tensor([[10, 11], [1, 2]])
print(x.shape)  # Exibe a forma do tensor: torch.Size([2, 2])

# Criação de um tensor 2D de 2x1 (coluna)
y = torch.tensor([[10], [11]])
print(y.shape)  # Exibe a forma do tensor: torch.Size([2, 1])

# Criação de um tensor 3D com dimensões 2x2x2
a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Extração dos valores x, y, z para cada elemento do tensor
x_vals = []
y_vals = []
z_vals = []
for i in range(a.shape[0]):  # Itera sobre a primeira dimensão
    for j in range(a.shape[1]):  # Itera sobre a segunda dimensão
        for k in range(a.shape[2]):  # Itera sobre a terceira dimensão
            x_vals.append(i)
            y_vals.append(j)
            z_vals.append(k)

# Criação de um gráfico 3D usando Plotly
trace = go.Scatter3d(
    x=x_vals,
    y=y_vals,
    z=z_vals,
    mode='markers',
    marker=dict(
        size=5,
        color=a.flatten(),  # Aplica os valores do tensor como cores
        colorscale='Viridis',
        opacity=0.8
    )
)

# Configuração do layout do gráfico 3D
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(title='Dimension 1'),
        yaxis=dict(title='Dimension 2'),
        zaxis=dict(title='Dimension 3')
    )
)

# Combinação do traço e layout em uma figura e exibição do gráfico
fig = go.Figure(data=[trace], layout=layout)
pyo.plot(fig, filename='tensor.html')

# Criação de tensores com números aleatórios
print(torch.zeros((3, 4)))  # Tensor 3x4 preenchido com zeros

# Tensor 3x4 com números inteiros aleatórios entre 0 e 10
print(torch.randint(low=0, high=10, size=(3, 4)))

# Tensor 3D 3x4x2 com números inteiros aleatórios entre 0 e 10
print(torch.randint(low=0, high=10, size=(3, 4, 2)))

# Tensor 3D 3x4x4 com números inteiros aleatórios entre 0 e 10
print(torch.randint(low=0, high=10, size=(3, 4, 4)))
