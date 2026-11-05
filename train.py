import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_network import NeuralNetwork
import os
import json

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

CLASSES = ["Setosa", "Versicolor", "Virginica"]
FEATURES = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]


def one_hot(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    for i, val in enumerate(y):
        encoded[i, val] = 1
    return encoded


def load_and_prepare_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    y_train_oh = one_hot(y_train, 3)
    y_test_oh = one_hot(y_test, 3)

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(os.path.join(STATIC_DIR, "scaler.json"), "w") as f:
        json.dump(scaler_params, f)

    return X_train, X_test, y_train, y_test, y_train_oh, y_test_oh, scaler, iris.data, iris.target


def plot_loss_curve(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss"], color="#e74c3c", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training Loss Over Epochs", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)


def plot_accuracy_curve(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["accuracy"], color="#2ecc71", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training Accuracy Over Epochs", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "accuracy_curve.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        annot_kws={"size": 16}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_decision_boundary(nn, X, y, scaler):
    feature_pairs = [(0, 1), (2, 3)]
    pair_names = [
        ("Sepal Length", "Sepal Width"),
        ("Petal Length", "Petal Width"),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (f1, f2) in enumerate(feature_pairs):
        ax = axes[idx]
        X_pair = X[:, [f1, f2]]

        scaler_pair = StandardScaler()
        X_pair_scaled = scaler_pair.fit_transform(X_pair)

        nn_pair = NeuralNetwork([2, 8, 3], learning_rate=0.05)
        y_oh = one_hot(y, 3)
        nn_pair.train(X_pair_scaled, y_oh, epochs=1500, verbose=False)

        x_min, x_max = X_pair_scaled[:, 0].min() - 1, X_pair_scaled[:, 0].max() + 1
        y_min, y_max = X_pair_scaled[:, 1].min() - 1, X_pair_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = nn_pair.predict_class(grid).reshape(xx.shape)

        ax.contourf(xx, yy, preds, alpha=0.3, colors=colors, levels=[-0.5, 0.5, 1.5, 2.5])
        for c in range(3):
            mask = y == c
            ax.scatter(
                X_pair_scaled[mask, 0], X_pair_scaled[mask, 1],
                c=colors[c], label=CLASSES[c], edgecolors="k", s=50, linewidth=0.5
            )
        ax.set_xlabel(pair_names[idx][0], fontsize=11)
        ax.set_ylabel(pair_names[idx][1], fontsize=11)
        ax.set_title(f"Decision Boundary: {pair_names[idx][0]} vs {pair_names[idx][1]}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "decision_boundary.png"), dpi=150)
    plt.close(fig)


def plot_network_architecture(layer_sizes):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    max_nodes = max(layer_sizes)
    ax.set_ylim(-0.5, max_nodes + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Neural Network Architecture", fontsize=16, fontweight="bold", pad=20)

    layer_labels = ["Input\nLayer"] + [f"Hidden\nLayer {i}" for i in range(1, len(layer_sizes) - 1)] + ["Output\nLayer"]
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    node_positions = []

    for l_idx, size in enumerate(layer_sizes):
        positions = []
        offset = (max_nodes - size) / 2
        for n_idx in range(size):
            x = l_idx
            y = offset + n_idx
            positions.append((x, y))

            if l_idx == 0:
                color = colors[0]
            elif l_idx == len(layer_sizes) - 1:
                color = colors[2]
            else:
                color = colors[1]

            circle = plt.Circle((x, y), 0.15, color=color, ec="black", linewidth=1.5, zorder=5)
            ax.add_patch(circle)

        node_positions.append(positions)
        ax.text(l_idx, -1, layer_labels[l_idx], ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(l_idx, -1.8, f"({size} nodes)", ha="center", va="center", fontsize=9, color="gray")

    for l_idx in range(len(layer_sizes) - 1):
        for start in node_positions[l_idx]:
            for end in node_positions[l_idx + 1]:
                ax.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color="gray", alpha=0.3, linewidth=0.8, zorder=1
                )

    input_labels = ["Sepal\nLength", "Sepal\nWidth", "Petal\nLength", "Petal\nWidth"]
    for i, (x, y) in enumerate(node_positions[0]):
        ax.text(x - 0.4, y, input_labels[i], ha="right", va="center", fontsize=8)

    output_labels = CLASSES
    for i, (x, y) in enumerate(node_positions[-1]):
        ax.text(x + 0.4, y, output_labels[i], ha="left", va="center", fontsize=8)

    ax.set_ylim(-2.5, max_nodes + 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "network_architecture.png"), dpi=150)
    plt.close(fig)


def plot_feature_distribution(X, y):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, ax in enumerate(axes.flat):
        for c in range(3):
            mask = y == c
            ax.hist(X[mask, i], bins=15, alpha=0.6, color=colors[c], label=CLASSES[c], edgecolor="white")
        ax.set_xlabel(FEATURES[i], fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Distribution of {FEATURES[i]}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle("Feature Distributions by Species", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC_DIR, "feature_distribution.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test, y_train_oh, y_test_oh, scaler, X_full, y_full = load_and_prepare_data()

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    layer_sizes = [4, 8, 8, 3]
    nn = NeuralNetwork(layer_sizes, learning_rate=0.05)

    print(f"\nNetwork architecture: {layer_sizes}")
    print("Training...\n")
    nn.train(X_train, y_train_oh, epochs=2000)

    y_pred = nn.predict_class(X_test)
    test_accuracy = np.mean(y_pred == y_test) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}%")

    nn.save_model(os.path.join(STATIC_DIR, "model.json"))

    stats = {
        "test_accuracy": round(float(test_accuracy), 1),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "epochs": 2000,
        "learning_rate": 0.05,
        "architecture": layer_sizes,
        "final_loss": round(nn.training_history["loss"][-1], 4),
        "final_train_accuracy": round(nn.training_history["accuracy"][-1], 1),
    }
    with open(os.path.join(STATIC_DIR, "stats.json"), "w") as f:
        json.dump(stats, f)

    print("\nGenerating visualizations...")
    plot_loss_curve(nn.training_history)
    print("  Loss curve saved.")

    plot_accuracy_curve(nn.training_history)
    print("  Accuracy curve saved.")

    plot_confusion_matrix(y_test, y_pred)
    print("  Confusion matrix saved.")

    plot_decision_boundary(nn, X_full, y_full, scaler)
    print("  Decision boundary saved.")

    plot_network_architecture(layer_sizes)
    print("  Network architecture saved.")

    plot_feature_distribution(X_full, y_full)
    print("  Feature distribution saved.")

    print("\nAll done! Run 'python app.py' to start the web app.")
