from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

# Fetch the MNIST dataset
mnist = fetch_mldata("MNIST original")
print("Successfully fetched MNIST dataset")

# Separate the dataset into training and testing
X, y = mnist.data / 255., mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14285, random_state=42)

import matplotlib.pyplot as plt
import numpy as np

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

bins = range(11)
plt.hist(y_test, bins=bins)
bins_labels(bins, fontsize=20)

plt.title("Testing Data Distribution")
plt.xlabel("Target")
plt.ylabel("Frequency")

plt.show()
