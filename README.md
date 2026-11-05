# Iris Flower Classifier

A Multilayer Perceptron (MLP) neural network built **from scratch** in Python (using only NumPy) to classify Iris flowers into three species: **Setosa**, **Versicolor**, and **Virginica**.

This project is part of the **Soft Computing and Applications (CSE320)** course.

---

## Features

- Neural network implemented from scratch with backpropagation (no TensorFlow/PyTorch/sklearn for the model)
- Web app with interactive sliders to classify flowers in real time
- 6 visualizations: loss curve, accuracy curve, confusion matrix, decision boundaries, network architecture, and feature distributions

---

## Project Structure

```
softcomputing/
├── neural_network.py      # MLP class built from scratch
├── train.py               # Trains the model and generates visualizations
├── app.py                 # Flask web app
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Web UI
└── static/                # Generated after training (charts + model files)
```

---

## How to Run (Step by Step)

### 1. Prerequisites

Make sure you have **Python 3.8+** installed on your laptop. You can check by running:

```bash
python --version
```

If not installed, download it from [python.org](https://www.python.org/downloads/) and install it. Make sure to check **"Add Python to PATH"** during installation.

### 2. Clone the Repository

Open a terminal (Command Prompt / PowerShell / Terminal) and run:

```bash
git clone https://github.com/aryan-py/softcomputing.git
```

```bash
cd softcomputing
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

This will train the neural network on the Iris dataset and generate all the visualization charts in the `static/` folder:

```bash
python train.py
```

You should see output like:

```
Loading Iris dataset...
Training set: 120 samples
Test set: 30 samples

Network architecture: [4, 8, 8, 3]
Training...

Epoch 100/2000 - Loss: 1.0588 - Accuracy: 51.7%
...
Epoch 2000/2000 - Loss: 0.1532 - Accuracy: 96.7%

Test Accuracy: 96.7%

Generating visualizations...
All done! Run 'python app.py' to start the web app.
```

### 5. Start the Web App

```bash
python app.py
```

### 6. Open in Browser

Open your browser and go to:

```
http://localhost:5000
```

You will see the Iris Flower Classifier with:
- Sliders to input flower measurements (sepal length, sepal width, petal length, petal width)
- A "Classify Flower" button that predicts the species with probability bars
- Training visualizations (loss curve, accuracy curve, confusion matrix, decision boundaries, network architecture, feature distributions)

---

## Neural Network Details

| Parameter | Value |
|-----------|-------|
| Architecture | 4 → 8 → 8 → 3 |
| Input Features | Sepal Length, Sepal Width, Petal Length, Petal Width |
| Output Classes | Setosa, Versicolor, Virginica |
| Activation (Hidden) | Sigmoid |
| Activation (Output) | Softmax |
| Loss Function | Cross-Entropy |
| Learning Rate | 0.05 |
| Epochs | 2000 |
| Test Accuracy | ~96.7% |

---

## Screenshots

After running the app, the web interface shows:

1. **Stats bar** with test accuracy, architecture, epochs, and final loss
2. **Prediction panel** with sliders and probability bars
3. **Network architecture** diagram
4. **Training loss curve** showing loss decreasing over epochs
5. **Training accuracy curve** showing accuracy improving
6. **Confusion matrix** on test data
7. **Decision boundary** plots for feature pairs
8. **Feature distribution** histograms by species
