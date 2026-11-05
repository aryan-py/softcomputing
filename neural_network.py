import numpy as np
import json
import os

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.training_history = {"loss": [], "accuracy": []}

        np.random.seed(42)
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        for i in range(self.num_layers - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            if i == self.num_layers - 2:
                a = self.softmax(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
        return activations

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def backpropagate(self, activations, y_true):
        m = y_true.shape[0]
        deltas = [None] * (self.num_layers - 1)

        deltas[-1] = activations[-1] - y_true

        for i in range(self.num_layers - 3, -1, -1):
            deltas[i] = (deltas[i + 1] @ self.weights[i + 1].T) * self.sigmoid_derivative(activations[i + 1])

        for i in range(self.num_layers - 1):
            dw = (activations[i].T @ deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            activations = self.forward(X)
            loss = self.cross_entropy_loss(activations[-1], y)

            predictions = np.argmax(activations[-1], axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels) * 100

            self.training_history["loss"].append(float(loss))
            self.training_history["accuracy"].append(float(accuracy))

            self.backpropagate(activations, y)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.1f}%")

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]

    def predict_class(self, X):
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

    def save_model(self, filepath):
        model_data = {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.learning_rate,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "training_history": self.training_history,
        }
        with open(filepath, "w") as f:
            json.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        nn = cls(data["layer_sizes"], data["learning_rate"])
        nn.weights = [np.array(w) for w in data["weights"]]
        nn.biases = [np.array(b) for b in data["biases"]]
        nn.training_history = data["training_history"]
        return nn
