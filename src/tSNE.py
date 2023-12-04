import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load a dataset for demonstration
digits = datasets.load_digits()
data = digits.data
target = digits.target

# Apply t-SNE to reduce dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(data)

# Visualize the 2D representation
plt.figure(figsize=(8, 6))
scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=target, cmap='viridis')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.title('t-SNE Visualization of Digits Dataset')
plt.show()
