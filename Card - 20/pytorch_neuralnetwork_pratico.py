import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

class SimpleNN(nn.Module):
    """
    Uma rede neural simples com duas camadas lineares:
    - Camada 1: 2 neurônios de entrada conectados a 3 neurônios na camada oculta.
    - Camada 2: 3 neurônios da camada oculta conectados a 1 neurônio de saída.
    """
    def __init__(self):
        """
        Inicializa as camadas da rede:
        - layer1: Conecta 2 neurônios de entrada a 3 neurônios na camada oculta.
        - layer2: Conecta 3 neurônios na camada oculta a 1 neurônio de saída.
        """
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 3)  # Camada de entrada para camada oculta
        self.layer2 = nn.Linear(3, 1)  # Camada oculta para camada de saída

    def forward(self, x):
        """
        Define a passagem dos dados pela rede.
        - Entrada: Vetor com 2 características.
        - Saída: Valor entre 0 e 1 (após a aplicação de sigmoide).
        """
        x = torch.sigmoid(self.layer1(x))  # Ativação sigmoide na camada oculta
        x = torch.sigmoid(self.layer2(x))  # Ativação sigmoide na saída
        return x

# Instancia o modelo e obtém os pesos das camadas
model = SimpleNN()
weights_layer1 = model.layer1.weight.detach().numpy()  # Pesos da primeira camada
weights_layer2 = model.layer2.weight.detach().numpy()  # Pesos da segunda camada

# Criação do grafo para visualizar a arquitetura da rede
graph = nx.DiGraph()

# Adiciona nós para as camadas (entrada, oculta, saída)
for i in range(2):  # 2 neurônios de entrada
    graph.add_node(f"Input_{i+1}", subset=0)
for j in range(3):  # 3 neurônios na camada oculta
    graph.add_node(f"Hidden_{j+1}", subset=1)
for k in range(1):  # 1 neurônio de saída
    graph.add_node(f"Output_{k+1}", subset=2)

# Adiciona arestas entre os nós com os pesos da primeira camada
for i in range(2):
    for j in range(3):
        graph.add_edge(f"Input_{i+1}", f"Hidden_{j+1}", weight=weights_layer1[j, i])

# Adiciona arestas entre os nós com os pesos da segunda camada
for j in range(3):
    for k in range(1):
        graph.add_edge(f"Hidden_{j+1}", f"Output_{k+1}", weight=weights_layer2[k, j])

# Define a posição dos nós para um layout multipartite
pos = nx.multipartite_layout(graph, subset_key="subset")

# Cria rótulos para as arestas com os valores dos pesos
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}

# Desenha o grafo da rede neural
nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="lightblue")
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

# Exibe o título e o gráfico
plt.title("Rede Neural com Pesos")
plt.show()
