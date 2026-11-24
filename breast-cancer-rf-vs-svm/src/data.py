"""Загрузка датасета Breast Cancer Wisconsin и разбиение train / val / test."""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42


def load_data():
    """Возвращает признаки X и цель y, где 1 = злокачественная опухоль."""
    data = load_breast_cancer(as_frame=True)  # датасет встроен в sklearn, скачивать не нужно
    X = data.data  # 30 числовых признаков клеточных ядер
    # В sklearn 0 = malignant, поэтому инвертируем: иначе Recall считался бы для доброкачественных
    y = (1 - data.target).rename("malignant")
    return X, y


def split_data(X, y, seed=SEED):
    """Делит данные 70/15/15 со stratify, чтобы доля класса совпадала во всех сплитах."""
    # Сначала откладываем 30% на val и test вместе
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # Затем делим отложенную часть пополам
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_summary(y_train, y_val, y_test):
    """Таблица для отчёта: размер каждого сплита и доля злокачественных опухолей."""
    rows = [("train", y_train), ("val", y_val), ("test", y_test)]
    return pd.DataFrame(
        [
            {"сплит": name, "строк": len(y), "злокачественных": int(y.sum()), "доля": round(y.mean(), 3)}
            for name, y in rows
        ]
    )
