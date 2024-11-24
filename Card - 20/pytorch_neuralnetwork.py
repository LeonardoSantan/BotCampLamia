import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 3)
        self.layer2 = nn.Linear(3, 1)
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

model = SimpleNN()
weights_layer1 = model.layer1.weight.detach().numpy()
weights_layer2 = model.layer2.weight.detach().numpy()

graph = nx.DiGraph()
for i in range(2):
    graph.add_node(f"Input_{i+1}", subset=0)
for j in range(3):
    graph.add_node(f"Hidden_{j+1}", subset=1)
for k in range(1):
    graph.add_node(f"Output_{k+1}", subset=2)

for i in range(2):
    for j in range(3):
        graph.add_edge(f"Input_{i+1}", f"Hidden_{j+1}", weight=weights_layer1[j, i])

for j in range(3):
    for k in range(1):
        graph.add_edge(f"Hidden_{j+1}", f"Output_{k+1}", weight=weights_layer2[k, j])

pos = nx.multipartite_layout(graph, subset_key="subset")
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="lightblue")
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
plt.title("Rede Neural com Pesos")
plt.show()
