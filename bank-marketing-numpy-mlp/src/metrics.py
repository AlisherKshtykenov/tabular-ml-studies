"""Метрики для дисбаланса, выбор порога, замер времени и бутстрап разности."""
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fit_timed(model, X, y):
    """Обучает модель и возвращает её вместе со временем обучения в секундах."""
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def proba(model, X):
    """Вероятность положительного класса: одинаковый интерфейс для всех моделей."""
    return model.predict_proba(X)[:, 1]


def best_threshold(y, p):
    """Порог с максимальным F1. Выбирается только на val, на тесте просто применяется."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def top_k_capture(y, p, share=0.2):
    """Какую долю всех согласий поймаем, если позвоним лучшим share клиентам по оценке модели."""
    y = np.asarray(y)
    k = int(len(y) * share)
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def evaluate(y, p, threshold):
    """Все метрики работы. PR-AUC и ROC-AUC от порога не зависят, остальные зависят."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Захват топ-20%": top_k_capture(y, p, 0.2),
        "Порог": threshold,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def bootstrap_pr_auc_diff(y, p_a, p_b, n_boot=2000, seed=42):
    """Доверительный интервал 95% для разности PR-AUC(a) - PR-AUC(b) на одних и тех же клиентах."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))  # одна выборка с возвращением для обеих моделей
        diffs.append(average_precision_score(y[idx], p_a[idx]) - average_precision_score(y[idx], p_b[idx]))
    return {
        "разность": float(average_precision_score(y, p_a) - average_precision_score(y, p_b)),
        "ДИ 2.5%": float(np.percentile(diffs, 2.5)),
        "ДИ 97.5%": float(np.percentile(diffs, 97.5)),
    }
