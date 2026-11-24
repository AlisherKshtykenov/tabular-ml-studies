"""Три архитектуры нейросетей на PyTorch. Все возвращают logit формы (batch,)."""
import torch
from torch import nn


class SimpleNN(nn.Module):
    """Простая сеть: один скрытый слой поверх one-hot входа."""

    def __init__(self, n_inputs, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit без сигмоиды: её внутри себя применит loss


class DeepNN(nn.Module):
    """Более сложная сеть: несколько слоёв, BatchNorm и Dropout. Вход тот же one-hot."""

    def __init__(self, n_inputs, hidden=(256, 128, 64, 32), dropout=0.2):
        super().__init__()
        layers, size = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            size = h
        layers.append(nn.Linear(size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """x -> x + F(x), где F = BatchNorm, Linear, ReLU, Dropout, Linear, Dropout."""

    def __init__(self, d, dropout, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.f = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, 2 * d),  # расширение даёт блоку место для комбинаций признаков
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),  # возврат к размеру d, чтобы можно было сложить с x
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.f(x) if self.use_skip else self.f(x)


def embedding_dim(cardinality):
    """Правило fastai: размер embedding растёт медленнее числа категорий."""
    return min(50, round(1.6 * cardinality**0.56))


class WideDeepResNet(nn.Module):
    """Своя сеть: logit = wide (линейная часть) + deep (embeddings и residual-блоки).

    Флаги use_wide, use_skip, learned_embeddings нужны для ablation: отключаем по одной части.
    """

    def __init__(self, n_num, cardinalities, d=128, n_blocks=3, dropout=0.2,
                 use_wide=True, use_skip=True, learned_embeddings=True):
        super().__init__()
        self.use_wide = use_wide
        self.learned_embeddings = learned_embeddings
        self.cardinalities = list(cardinalities)

        if learned_embeddings:
            # У каждой категории свой обучаемый вектор: сеть сама решает, какие категории похожи
            self.embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim(c)) for c in cardinalities])
            deep_in = n_num + sum(embedding_dim(c) for c in cardinalities)
        else:
            deep_in = n_num + sum(cardinalities)  # вариант для ablation: фиксированный one-hot

        self.input_layer = nn.Linear(deep_in, d)  # приводим смесь признаков к рабочему размеру d
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[ResidualBlock(d, dropout, use_skip) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, 1))

        if use_wide:
            # Wide-часть это логистическая регрессия: вес каждого числа и каждой категории
            self.wide_num = nn.Linear(n_num, 1)
            self.wide_cat = nn.ModuleList([nn.Embedding(c, 1) for c in cardinalities])

    def _deep_input(self, x_num, x_cat):
        """Склеивает числа и векторы категорий в один вход deep-части."""
        if self.learned_embeddings:
            parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        else:
            parts = [
                nn.functional.one_hot(x_cat[:, i], c).float() for i, c in enumerate(self.cardinalities)
            ]
        return torch.cat([x_num, *parts], dim=1)

    def forward(self, x_num, x_cat):
        h = self.input_dropout(self.input_layer(self._deep_input(x_num, x_cat)))
        logit = self.head(self.blocks(h)).squeeze(-1)
        if self.use_wide:
            logit = logit + self.wide_num(x_num).squeeze(-1)  # прямой путь от входа к выходу
            for i, emb in enumerate(self.wide_cat):
                logit = logit + emb(x_cat[:, i]).squeeze(-1)
        return logit


def count_parameters(model):
    """Число обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
