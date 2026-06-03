"""
layers.py
---------
Reusable deep learning building blocks built from scratch.

Classes:
    Perceptron: Single neuron classifier
    NeuralNetwork: 2-layer neural network
    VanillaRNN: Recurrent neural network
    SelfAttention: Transformer self-attention
"""

import numpy as np


# --- Activation Functions ---

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation — squashes values between 0 and 1."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid — needed for backpropagation."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation — max(0, x)."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# --- Layers ---

class Perceptron:
    """
    Single neuron classifier.
    Learns a linear decision boundary using the perceptron learning rule.

    Limitation: cannot solve non-linearly separable problems (e.g. XOR).
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []

    def _activation(self, x: float) -> int:
        return 1 if x >= 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            idx = np.random.permutation(n_samples)
            errors = 0
            for i in idx:
                prediction = self._activation(
                    np.dot(X[i], self.weights) + self.bias
                )
                update = self.learning_rate * (y[i] - prediction)
                self.weights += update * X[i]
                self.bias += update
                errors += int(update != 0)
            self.errors.append(errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self._activation(x) for x in linear_output])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class NeuralNetwork:
    """
    2-layer neural network.
    Architecture: input -> hidden -> output
    Uses sigmoid activation and MSE loss.

    Solves non-linearly separable problems (e.g. XOR) using hidden layers.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, learning_rate: float = 0.5):
        self.lr = learning_rate
        self.losses = []

        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        m = X.shape[0]
        dL_da2 = self.a2 - y.reshape(-1, 1)
        dL_dz2 = dL_da2 * sigmoid_derivative(self.z2)
        dL_dW2 = np.dot(self.a1.T, dL_dz2) / m
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * sigmoid_derivative(self.z1)
        dL_dW1 = np.dot(X.T, dL_dz1) / m
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000) -> None:
        for _ in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            self.losses.append(loss)
            self.backward(X, y)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(X) >= threshold).astype(int).flatten()

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


class VanillaRNN:
    """
    Simple recurrent neural network.
    Processes sequences one step at a time using a hidden state.

    Limitation: vanishing gradient over long sequences.
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        np.random.seed(42)
        self.W_x = np.random.randn(hidden_size, input_size) * 0.1
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b = np.zeros((hidden_size, 1))

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        x_t = x_t.reshape(self.W_x.shape[1], 1)
        return np.tanh(self.W_x @ x_t + self.W_h @ h_prev + self.b)

    def forward(self, sequence: np.ndarray) -> list:
        h = np.zeros((self.hidden_size, 1))
        hidden_states = []
        for x_t in sequence:
            h = self.step(x_t, h)
            hidden_states.append(h.copy())
        return hidden_states


class SelfAttention:
    """
    Scaled dot-product self-attention.
    Core mechanism of the transformer architecture.

    Every token attends to every other token simultaneously.
    No sequential bottleneck — no vanishing gradient across positions.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model
        np.random.seed(42)
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X: np.ndarray) -> tuple:
        """
        Args:
            X: Input embeddings (seq_len, d_model)

        Returns:
            output: Attended values (seq_len, d_model)
            weights: Attention weights (seq_len, seq_len)
        """
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        scores = Q @ K.T / np.sqrt(self.d_model)
        weights = softmax(scores)
        output = weights @ V

        return output, weights


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding.
    Adds position information to token embeddings.
    """
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE