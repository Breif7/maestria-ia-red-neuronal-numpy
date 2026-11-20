import numpy as np

from mlp import MLP, numeric_grad_error, softmax, spirals, split


def test_backprop_matches_numeric_gradient():
    X, y = spirals(n_per_class=5)
    assert numeric_grad_error(MLP([2, 6, 3], seed=1), X, y) < 1e-5


def test_softmax_is_stable_and_normalised():
    p = softmax(np.array([[1000.0, 1000.0, -1000.0]]))
    assert np.allclose(p.sum(), 1) and np.allclose(p[0, :2], 0.5)


def test_loss_decreases_during_training():
    X, y = spirals(n_per_class=30)
    hist = MLP([2, 32, 3]).fit(X, y, epochs=200)
    assert hist[-1] < hist[0] * 0.5


def test_hidden_layer_beats_linear_on_spirals():
    Xtr, ytr, Xte, yte = split(*spirals())
    lin, mlp = MLP([2, 3]), MLP([2, 64, 3])
    lin.fit(Xtr, ytr, epochs=1500)
    mlp.fit(Xtr, ytr, epochs=1500)
    acc_lin = np.mean(lin.predict(Xte) == yte)
    acc_mlp = np.mean(mlp.predict(Xte) == yte)
    assert acc_lin < 0.7 and acc_mlp > 0.9


def test_same_seed_same_result():
    X, y = spirals(n_per_class=20)
    a, b = MLP([2, 8, 3], seed=3), MLP([2, 8, 3], seed=3)
    assert a.fit(X, y, epochs=20) == b.fit(X, y, epochs=20)
