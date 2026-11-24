"""Цикл обучения нейросети и параллельный запуск экспериментов."""
import copy
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from torch import nn

from . import nn_models


def to_tensors(arrays):
    """Копирует массивы в тензоры: joblib и pandas отдают массивы только для чтения."""
    return [torch.from_numpy(np.array(a)) for a in arrays]


def to_target(y):
    """Цель как float-тензор: этого требует BCEWithLogitsLoss."""
    return torch.from_numpy(np.array(y, dtype=np.float32))


def build_model(kind, params):
    """Создаёт модель по имени. Нужно, чтобы конфиг можно было передать в другой процесс."""
    classes = {"simple": nn_models.SimpleNN, "deep": nn_models.DeepNN, "own": nn_models.WideDeepResNet}
    return classes[kind](**params)


def predict_logits(model, inputs, batch_size=4096):
    """Считает logit частями в режиме eval: Dropout выключен, BatchNorm на накопленной статистике."""
    model.eval()
    with torch.no_grad():
        parts = [
            model(*[x[i : i + batch_size] for x in inputs]) for i in range(0, len(inputs[0]), batch_size)
        ]
    return torch.cat(parts)


def fit(kind, params, train_inputs, y_train, val_inputs, y_val, lr=1e-3, weight_decay=1e-4,
        batch_size=256, max_epochs=60, patience=8, seed=42):
    """Обучает модель mini-batch градиентным спуском с ранней остановкой по val loss."""
    torch.manual_seed(seed)  # одинаковая инициализация и порядок батчей при одном seed
    torch.set_num_threads(1)  # параллелим запуски целиком, а не вычисления внутри запуска
    model = build_model(kind, params)
    train_inputs, val_inputs = to_tensors(train_inputs), to_tensors(val_inputs)
    y_tr, y_va = to_target(y_train), to_target(y_val)

    criterion = nn.BCEWithLogitsLoss()  # sigmoid + log loss, численно устойчивее ручной сигмоиды
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Если val loss не падает 3 эпохи, шаг обучения уменьшается вдвое
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()  # Dropout активен, BatchNorm обновляет статистику
        order = torch.randperm(len(y_tr))  # новый порядок объектов каждую эпоху
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            if len(idx) < 2:  # BatchNorm не работает на батче из одного объекта
                continue
            optimizer.zero_grad()  # обнуляем градиенты прошлого шага
            loss = criterion(model(*[x[idx] for x in train_inputs]), y_tr[idx])  # прямой проход
            loss.backward()  # backpropagation: градиент loss по всем весам
            optimizer.step()  # обновляем веса
            total += loss.item() * len(idx)
        val_logits = predict_logits(model, val_inputs)
        val_loss = criterion(val_logits, y_va).item()
        val_pr = average_precision_score(y_val, val_logits.numpy())
        history.append(
            {"epoch": epoch, "train_loss": total / len(y_tr), "val_loss": val_loss, "val_pr_auc": val_pr,
             "lr": optimizer.param_groups[0]["lr"]}
        )
        scheduler.step(val_loss)
        if val_loss < best_loss - 1e-5:  # есть улучшение: запоминаем веса этой эпохи
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:  # долго нет улучшения: дальше только переобучение
                break
    model.load_state_dict(best_state)  # возвращаем лучшие веса, а не последние
    return {
        "kind": kind, "params": params, "state": best_state, "history": history, "best_epoch": best_epoch,
        "best_val_loss": best_loss, "train_time": time.perf_counter() - start,
        "val_scores": predict_logits(model, val_inputs).numpy(), "seed": seed,
    }


def restore(run):
    """Собирает обученную модель из результата fit."""
    model = build_model(run["kind"], run["params"])
    model.load_state_dict(run["state"])
    return model.eval()


def run_many(configs, n_jobs=8):
    """Запускает список конфигов параллельно. Результат не зависит от порядка: у каждого свой seed."""
    return Parallel(n_jobs=n_jobs)(delayed(fit)(**c) for c in configs)


def overfit_check(kind, params, inputs, y, steps=500, lr=3e-3, seed=42):
    """Проверка кода: исправная сеть обязана выучить маленький батч почти до нулевого loss."""
    torch.manual_seed(seed)
    model = build_model(kind, params)
    # Dropout мешает выучить батч наизусть, поэтому на этой проверке он выключен
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    x, t = to_tensors(inputs), to_target(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    start_loss = None
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(*x), t)
        loss.backward()
        optimizer.step()
        start_loss = loss.item() if start_loss is None else start_loss
    return start_loss, loss.item()


def first_layer_grad_norm(n_num, cardinalities, depth, use_skip, x_num, x_cat, y, seed=42):
    """Норма градиента на самом первом слое до обучения: сколько сигнала доходит до входа."""
    torch.manual_seed(seed)
    model = nn_models.WideDeepResNet(
        n_num, cardinalities, d=128, n_blocks=depth, dropout=0.0, use_wide=False, use_skip=use_skip
    )
    model.train()
    loss = nn.BCEWithLogitsLoss()(model(*to_tensors([x_num, x_cat])), to_target(y))
    loss.backward()
    return float(np.linalg.norm(model.input_layer.weight.grad.numpy()))
