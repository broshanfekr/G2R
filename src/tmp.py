import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

cov1 = np.array([[10, -3],
                 [-3, 2.5]])
pts1 = np.random.multivariate_normal([5, 0], cov1, size=100)

cov2 = np.array([[10, 3],
                 [3, 2.5]])
pts2 = np.random.multivariate_normal([-5, 0], cov2, size=100)

cov3 = np.array([[10, 0],
                 [0, 2.5]])
pts3 = np.random.multivariate_normal([0, -5], cov3, size=100)

pts = np.concatenate([pts1, pts2, pts3])

plt.plot(pts1[:, 0], pts1[:, 1], '.', alpha=0.5, c='red')
plt.plot(pts2[:, 0], pts2[:, 1], '.', alpha=0.5, c='blue')
plt.plot(pts3[:, 0], pts3[:, 1], '.', alpha=0.5, c='green')
plt.axis('equal')
plt.grid()
plt.show()


in_prob = 0.7
out_prob = 0.1

sizes = [100, 100, 100]
probs = [[in_prob, out_prob, out_prob], [out_prob, in_prob, out_prob], [out_prob, out_prob, in_prob]]

G = nx.stochastic_block_model(sizes, probs, seed=0)

# pos = nx.spectral_layout(G)
pos = pts

colors = []
c_idx = 0
for b in sizes:
    for i in range(b):
        colors.append(c_idx)
    c_idx += 1


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