"""Загрузка Bank Marketing (UCI) и разбиения: случайное 70/15/15 и по времени."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LEAKY_COLUMN = "duration"  # длительность звонка известна только после звонка, это утечка


def load_raw(root):
    """Читает исходный файл без изменений. Порядок строк совпадает с порядком дат."""
    return pd.read_csv(Path(root) / "data" / "raw" / "bank-full.csv", sep=";")


def load_data(root):
    """Возвращает признаки X без утечки и цель y, где 1 = клиент открыл депозит."""
    df = load_raw(root)
    y = (df["y"] == "yes").astype(int).rename("subscribed")
    X = df.drop(columns=["y", LEAKY_COLUMN])
    return X, y


def split_data(X, y, seed=SEED):
    """Делит данные 70/15/15 со stratify: доля класса одинакова во всех сплитах."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def temporal_split(X, y, test_fraction=0.15):
    """Обучение на ранних строках, проверка на поздних: так модель работала бы в реальности."""
    cut = int(len(X) * (1 - test_fraction))  # строки уже идут по порядку дат
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def split_summary(splits):
    """Таблица для отчёта: размер сплита и доля положительного класса."""
    return pd.DataFrame(
        [
            {"сплит": name, "строк": len(y), "согласий": int(y.sum()), "доля": round(float(y.mean()), 3)}
            for name, y in splits.items()
        ]
    )
