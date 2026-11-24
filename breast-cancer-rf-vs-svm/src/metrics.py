"""Метрики качества, стоимость ошибок и замер времени обучения."""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

COST_FP = 1  # ложная тревога: лишнее обследование здорового пациента
COST_FN = 20  # пропущенная опухоль: цена ошибки в 20 раз выше


def fit_timed(model, X, y, repeats=1):
    """Обучает модель repeats раз и возвращает её вместе с медианным временем в секундах."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - start)
    return model, float(np.median(times))  # медиана устойчива к случайным задержкам системы


def scores(model, X):
    """Непрерывная оценка для ROC-AUC: у SVM это decision_function, у леса вероятность."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def evaluate(model, X, y):
    """Все метрики для положительного класса 1 = злокачественная опухоль."""
    pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y, scores(model, X)),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Cost": int(COST_FP * fp + COST_FN * fn),
    }


def repeated_cv(models, X, y, n_repeats=10, n_splits=5):
    """Повторная кросс-валидация: показывает разброс, который скрыт в одном разбиении."""
    rows = []
    for name, model in models.items():
        for repeat in range(n_repeats):
            # Новое перемешивание в каждом повторе даёт другие фолды
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
            # clone обучает чистую копию, предсказание для строки делает модель, не видевшая её
            pred = cross_val_predict(clone(model), X, y, cv=cv)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "Модель": name,
                    "Recall": tp / (tp + fn),
                    "Precision": tp / (tp + fp),
                    "Cost": COST_FP * fp + COST_FN * fn,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_cost_diff(y, pred_a, pred_b, n_boot=5000, seed=42):
    """Доверительный интервал 95% для разности Cost(a) - Cost(b) на одной выборке."""
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    # Стоимость ошибки на каждом пациенте: 0 если верно, иначе цена FP или FN
    weights = np.where(y == 1, COST_FN, COST_FP)
    cost_a = (pred_a != y) * weights
    cost_b = (pred_b != y) * weights
    # Выбираем пациентов с возвращением, модели оцениваются на одних и тех же пациентах
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = cost_a[idx].sum(axis=1) - cost_b[idx].sum(axis=1)
    return {
        "разность": int(cost_a.sum() - cost_b.sum()),
        "ДИ 2.5%": float(np.percentile(diffs, 2.5)),
        "ДИ 97.5%": float(np.percentile(diffs, 97.5)),
    }
