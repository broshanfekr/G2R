from cdlib import algorithms
import networkx as nx
import matplotlib.pyplot as plt

in_prob = 0.7
out_prob = 0.1

sizes = [10, 10, 10]
probs = [[in_prob, out_prob, out_prob], [out_prob, in_prob, out_prob], [out_prob, out_prob, in_prob]]
G = nx.stochastic_block_model(sizes, probs, seed=0)

pos = nx.spectral_layout(G)

colors = []
c_idx = 0
for b in sizes:
    for i in range(b):
        colors.append(c_idx)
    c_idx += 1




# coms = algorithms.conga(G, number_communities=3)
# coms = algorithms.congo(G, number_communities=3, height=2)
coms = algorithms.core_expansion(G)
# coms = algorithms.demon(G, min_com_size=3, epsilon=0.7)
# coms = algorithms.lais2(G)
# coms = algorithms.lfm(G, alpha=0.9)
# coms = algorithms.mnmf(G)


c_colors = [0 for i in range(len(colors))]
c = list(coms.communities)
print("communities are:")
for i, c_list in enumerate(c):
    print(i, c_list)
    for el in c_list:
        c_colors[el] = (i+1)
print("end of communites")


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

plt.savefig("GMCR2.png")

print("the end")

