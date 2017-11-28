from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import time

# Fetch the MNIST dataset
mnist = fetch_mldata("MNIST original")
print("Successfully fetched MNIST dataset")

# Separate the dataset into training and testing
X, y = mnist.data / 255., mnist.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14285, random_state=42)

start = time.time()

# Build the feedforward neural network
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=100, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=0.1)

# Train the feedforward neural network
mlp.fit(X_train, y_train)

# Measure the training and testing accuracy scores
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

end = time.time()
print("Execution time: %f" % (end - start))
