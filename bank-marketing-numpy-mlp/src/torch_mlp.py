"""Та же архитектура на PyTorch: контроль своей сети (контрольная реализация)."""
import numpy as np
import torch
from torch import nn


class TorchMLP:
    """Повторяет NumpyMLP: слои, Adam, L2, early stopping. Backprop делает autograd."""

    def __init__(self, hidden_layers=(64, 32), learning_rate=1e-3, l2=1e-4, batch_size=256,
                 max_epochs=100, patience=10, val_fraction=0.1, random_state=42):
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        X_all = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_all = torch.tensor(np.asarray(y), dtype=torch.float32)

        # Сеть: Linear + ReLU для каждого скрытого слоя, в конце один выход без активации
        layers, n_in = [], X_all.shape[1]
        for n_out in self.hidden_layers:
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            n_in = n_out
        layers.append(nn.Linear(n_in, 1))
        self.net_ = nn.Sequential(*layers)

        # L2 только на веса, как в NumpyMLP (у параметров сдвига размерность 1)
        decay = [p for p in self.net_.parameters() if p.dim() > 1]
        no_decay = [p for p in self.net_.parameters() if p.dim() == 1]
        optimizer = torch.optim.Adam(
            [{"params": decay, "weight_decay": self.l2}, {"params": no_decay, "weight_decay": 0.0}],
            lr=self.learning_rate,
        )
        loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + кросс-энтропия, численно устойчиво

        # Тот же внутренний сплит для early stopping
        val_parts = []
        y_np = np.asarray(y)
        for cls in (0, 1):
            idx = rng.permutation(np.flatnonzero(y_np == cls))
            val_parts.append(idx[: int(len(idx) * self.val_fraction)])
        val_idx = np.concatenate(val_parts)
        fit_idx = np.setdiff1d(np.arange(len(y_np)), val_idx)

        best_loss, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net_.train()
            order = rng.permutation(fit_idx)
            for start in range(0, len(order), self.batch_size):
                batch = torch.as_tensor(order[start : start + self.batch_size])
                optimizer.zero_grad()  # обнуляем градиенты прошлого шага
                loss = loss_fn(self.net_(X_all[batch]).squeeze(1), y_all[batch])
                loss.backward()  # backprop считает autograd
                optimizer.step()
            self.net_.eval()
            with torch.no_grad():
                val_tensor = torch.as_tensor(val_idx)
                val_loss = loss_fn(self.net_(X_all[val_tensor]).squeeze(1), y_all[val_tensor]).item()
            if val_loss < best_loss - 1e-5:
                best_loss, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(np.asarray(X), dtype=torch.float32)).squeeze(1)
            p = torch.sigmoid(logits).numpy().astype(float)
        return np.column_stack([1 - p, p])
