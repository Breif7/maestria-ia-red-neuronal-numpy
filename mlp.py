"""Multilayer perceptron written with NumPy only: forward pass, backprop, Adam.

Compared against a linear softmax classifier (no hidden layer) on a 3-class
spiral, a problem a straight decision boundary cannot solve.
"""
from __future__ import annotations

import numpy as np


def spirals(n_per_class: int = 100, classes: int = 3, noise: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, k))
    return np.vstack(X), np.concatenate(y)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # stability: avoid exp overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """sizes=[2, 3] is plain softmax regression; [2, 64, 3] adds one ReLU layer."""

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]  # He init
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(softmax(z) if i == len(self.W) - 1 else np.maximum(z, 0))
        return acts

    def loss_and_grads(self, X, y, l2: float = 0.0):
        acts = self.forward(X)
        n = len(y)
        p = acts[-1]
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean() + 0.5 * l2 * sum((W**2).sum() for W in self.W)
        delta = p.copy()
        delta[np.arange(n), y] -= 1  # d(cross-entropy)/d(logits) for softmax
        delta /= n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + l2 * self.W[i]
            gb[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)
        return loss, gW, gb

    def fit(self, X, y, epochs: int = 500, lr: float = 0.01, l2: float = 1e-4):
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        for t in range(1, epochs + 1):
            loss, gW, gb = self.loss_and_grads(X, y, l2)
            history.append(float(loss))
            for i, (p, g) in enumerate(zip(params, gW + gb)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                p -= lr * (m[i] / (1 - b1**t)) / (np.sqrt(v[i] / (1 - b2**t)) + eps)
        return history

    def predict(self, X):
        return self.forward(X)[-1].argmax(axis=1)


def numeric_grad_error(model: MLP, X, y, l2: float = 1e-3, h: float = 1e-5) -> float:
    """Max relative error between backprop and central differences."""
    _, gW, _ = model.loss_and_grads(X, y, l2)
    worst = 0.0
    for W, g in zip(model.W, gW):
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + h
            lp = model.loss_and_grads(X, y, l2)[0]
            W[idx] = old - h
            lm = model.loss_and_grads(X, y, l2)[0]
            W[idx] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))
    return worst


def split(X, y, test_frac: float = 0.3, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(len(y))
    cut = int(len(y) * (1 - test_frac))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


if __name__ == "__main__":
    X, y = spirals()
    Xtr, ytr, Xte, yte = split(X, y)
    print(f"{'Modelo':<28}{'Exactitud prueba':>18}")
    print(f"{'Siempre la clase mayoritaria':<28}{np.mean(yte == np.bincount(ytr).argmax()):>18.3f}")
    for name, sizes in [("Softmax lineal [2,3]", [2, 3]), ("MLP [2,64,3]", [2, 64, 3]),
                        ("MLP [2,64,64,3]", [2, 64, 64, 3])]:
        model = MLP(sizes)
        model.fit(Xtr, ytr, epochs=2000)
        print(f"{name:<28}{np.mean(model.predict(Xte) == yte):>18.3f}")
