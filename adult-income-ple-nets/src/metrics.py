"""Метрики, выбор порога и bootstrap для сравнения моделей."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def best_threshold(y_true, scores):
    """Порог с максимальным F1. Подбирается только на val, на test применяется как есть."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)  # защита от деления на ноль
    return float(thresholds[np.argmax(f1[:-1])])  # у последней точки кривой порога нет


def evaluate(y_true, scores, threshold):
    """Метрики качества ранжирования (не зависят от порога) и классификации (зависят)."""
    y_pred = (scores >= threshold).astype(int)
    return {
        "PR-AUC": average_precision_score(y_true, scores),
        "ROC-AUC": roc_auc_score(y_true, scores),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def bootstrap_diff(y_true, scores_a, scores_b, metric=average_precision_score, n=2000, seed=42):
    """Парный bootstrap разницы метрик A минус B на одних и тех же объектах теста."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))  # выборка с возвращением
        diffs[i] = metric(y_true[idx], scores_a[idx]) - metric(y_true[idx], scores_b[idx])
    return {
        "разница": metric(y_true, scores_a) - metric(y_true, scores_b),
        "ci_low": np.percentile(diffs, 2.5),
        "ci_high": np.percentile(diffs, 97.5),
    }


def subgroup_table(y_true, y_pred, groups):
    """Качество по подгруппам: одна общая метрика может скрывать разную цену ошибок."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for g in pd.unique(groups):
        mask = np.asarray(groups) == g
        yt, yp = y_true[mask], y_pred[mask]
        rows.append(
            {
                "группа": g,
                "строк": int(mask.sum()),
                "доля >50K": yt.mean(),
                "предсказано >50K": yp.mean(),
                "Recall": yp[yt == 1].mean(),
                "FPR": yp[yt == 0].mean(),  # доля ложных срабатываний среди людей с доходом до 50K
                "Accuracy": (yt == yp).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("строк", ascending=False)
