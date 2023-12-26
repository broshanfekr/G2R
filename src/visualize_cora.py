import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()

suse = np.load("cora.npz")

adj = np.load("cora.npy")
ys = np.load("cora_y.npy")

G = nx.from_numpy_matrix(adj)
pos = nx.spectral_layout(G)
# pos = nx.spring_layout(G)
save_var("cora_pos.pckl", pos)

colors = list(ys)

c_colors = [0 for i in range(len(colors))]
c = list(nx.community.k_clique_communities(G, 5))
for i, c_list in enumerate(c):
    print(i, c_list)
    for el in c_list:
        c_colors[el] = (i+1)

no_label = [i for i in range(len(colors)) if c_colors[i] == 0]
print("no label: ", no_label)

fig, (ax1, ax2) = plt.subplots(1, 2)

nx.draw(G, pos=pos,
        ax=ax1,
        with_labels=True,
        font_weight='bold',
        font_color='black',
        font_size=10,
        node_color=colors,
        edge_color='gray',
        linewidths=1,
        alpha=0.7)
ax1.set_title('True partitions')

nx.draw(G, pos=pos,
        ax=ax2,
        with_labels=True,
        font_weight='bold',
        font_color='black',
        font_size=10,
        node_color=c_colors,
        edge_color='gray',
        linewidths=1,
        alpha=0.7)
ax2.set_title('predicted partitions')

plt.show()

print("the end")
