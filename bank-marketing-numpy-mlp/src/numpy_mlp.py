"""Своя нейросеть для бинарной классификации. Только numpy, никаких других библиотек."""
import numpy as np


class NumpyMLP:
    """Многослойный перцептрон: скрытые слои с ReLU или tanh, выход один (логит)."""

    def __init__(
        self,
        hidden_layers=(64, 32),
        activation="relu",
        learning_rate=1e-3,
        l2=1e-4,
        batch_size=256,
        max_epochs=100,
        patience=10,
        val_fraction=0.1,
        random_state=42,
    ):
        if activation not in ("relu", "tanh"):
            raise ValueError("activation должен быть relu или tanh")
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.learning_rate = learning_rate
        self.l2 = l2  # сила L2-регуляризации весов
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience  # сколько эпох ждать улучшения перед остановкой
        self.val_fraction = val_fraction
        self.random_state = random_state

    # ---------- инициализация ----------

    def _init_params(self, n_features, rng):
        """Создаёт веса W и сдвиги b каждого слоя."""
        sizes = (n_features, *self.hidden_layers, 1)
        self.W_, self.b_ = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He для ReLU и Xavier для tanh: дисперсия сигнала не гаснет от слоя к слою
            gain = 2.0 if self.activation == "relu" else 1.0
            self.W_.append(rng.normal(0.0, np.sqrt(gain / n_in), size=(n_in, n_out)))
            self.b_.append(np.zeros(n_out))
        # Память Adam: первый и второй моменты градиента для каждого параметра
        self._m = [np.zeros_like(p) for p in self.W_ + self.b_]
        self._v = [np.zeros_like(p) for p in self.W_ + self.b_]
        self._t = 0

    # ---------- функции активации ----------

    def _act(self, z):
        return np.maximum(z, 0.0) if self.activation == "relu" else np.tanh(z)

    def _act_grad(self, z):
        return (z > 0).astype(float) if self.activation == "relu" else 1.0 - np.tanh(z) ** 2

    # ---------- прямой проход ----------

    def _forward(self, X):
        """Считает логит и запоминает значения слоёв: они нужны обратному проходу."""
        a = X
        cache = []
        for layer, (W, b) in enumerate(zip(self.W_, self.b_)):
            z = a @ W + b  # линейная часть слоя
            cache.append((a, z))  # вход слоя и его предактивация
            # На последнем слое активации нет: sigmoid учтён внутри функции потерь
            a = z if layer == len(self.W_) - 1 else self._act(z)
        return a.ravel(), cache

    # ---------- функция потерь ----------

    def _loss(self, logits, y):
        """Бинарная кросс-энтропия через логит: не даёт log(0) при крайних значениях."""
        bce = np.maximum(logits, 0) - logits * y + np.logaddexp(0.0, -np.abs(logits))
        penalty = 0.5 * self.l2 * sum(np.sum(W**2) for W in self.W_)
        return float(bce.mean() + penalty)

    # ---------- обратный проход ----------

    def _backward(self, logits, y, cache):
        """Backpropagation: цепное правило от выхода к входу, возвращает все градиенты."""
        n = len(y)
        # dL/dz выходного слоя равна sigmoid(z) - y, это единственный «сигнал ошибки»
        delta = (self._sigmoid(logits) - y).reshape(-1, 1)
        grads_W, grads_b = [None] * len(self.W_), [None] * len(self.b_)
        for layer in reversed(range(len(self.W_))):
            a_in, z = cache[layer]
            if layer < len(self.W_) - 1:
                delta = delta * self._act_grad(z)  # переходим через активацию этого слоя
            grads_W[layer] = a_in.T @ delta / n + self.l2 * self.W_[layer]  # L2 только для весов
            grads_b[layer] = delta.mean(axis=0)
            delta = delta @ self.W_[layer].T  # передаём ошибку на выход предыдущего слоя
        return grads_W, grads_b

    @staticmethod
    def _sigmoid(z):
        # clip защищает exp от переполнения при очень больших логитах
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # ---------- оптимизатор Adam ----------

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self.W_ + self.b_
        for i, (param, grad) in enumerate(zip(params, grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * grad  # среднее градиента, инерция
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * grad**2  # среднее квадрата, масштаб
            m_hat = self._m[i] / (1 - beta1**self._t)  # поправка: в начале средние занижены
            v_hat = self._v[i] / (1 - beta2**self._t)
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)  # меняем массив на месте

    # ---------- обучение ----------

    def _internal_split(self, y, rng):
        """Отделяет часть train для early stopping с сохранением доли классов."""
        val_parts = []
        for cls in (0, 1):
            idx = rng.permutation(np.flatnonzero(y == cls))
            val_parts.append(idx[: int(len(idx) * self.val_fraction)])
        val_idx = np.concatenate(val_parts)
        return np.setdiff1d(np.arange(len(y)), val_idx), val_idx

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self._init_params(X.shape[1], rng)
        fit_idx, val_idx = self._internal_split(y, rng)
        self.history_ = {"train_loss": [], "val_loss": []}
        best_loss, best_params, wait = np.inf, None, 0

        for _ in range(self.max_epochs):
            order = rng.permutation(fit_idx)  # новый порядок объектов в каждой эпохе
            batch_losses = []
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                logits, cache = self._forward(X[batch])
                batch_losses.append(self._loss(logits, y[batch]))
                grads_W, grads_b = self._backward(logits, y[batch], cache)
                self._adam_step(grads_W + grads_b)

            val_logits, _ = self._forward(X[val_idx])
            val_loss = self._loss(val_logits, y[val_idx])
            self.history_["train_loss"].append(float(np.mean(batch_losses)))
            self.history_["val_loss"].append(val_loss)

            # Early stopping: запоминаем лучшие веса и останавливаемся, когда улучшений нет
            if val_loss < best_loss - 1e-5:
                best_loss, wait = val_loss, 0
                best_params = ([W.copy() for W in self.W_], [b.copy() for b in self.b_])
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.W_, self.b_ = best_params
        self.best_epoch_ = len(self.history_["val_loss"]) - wait
        return self

    # ---------- предсказание ----------

    def predict_proba(self, X):
        """Возвращает два столбца, как в sklearn: P(класс 0) и P(класс 1)."""
        logits, _ = self._forward(np.asarray(X, dtype=float))
        p = self._sigmoid(logits)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def gradient_check(model, X, y, eps=1e-6):
    """Сверяет backprop с численной производной: ||g - n|| / (||g|| + ||n||). Меньше 1e-6 значит, что код верный."""
    logits, cache = model._forward(X)
    grads_W, grads_b = model._backward(logits, y, cache)
    analytic, numeric = [], []
    for params, grads in ((model.W_, grads_W), (model.b_, grads_b)):
        for param, grad in zip(params, grads):
            for idx in np.ndindex(param.shape):
                original = param[idx]
                param[idx] = original + eps
                loss_plus = model._loss(model._forward(X)[0], y)
                param[idx] = original - eps
                loss_minus = model._loss(model._forward(X)[0], y)
                param[idx] = original  # обязательно возвращаем значение обратно
                analytic.append(grad[idx])
                numeric.append((loss_plus - loss_minus) / (2 * eps))
    analytic, numeric = np.array(analytic), np.array(numeric)
    return float(np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric)))
