"""Загрузка Adult Census, очистка, новые признаки и разбиение train / val / test."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "income_high"

# Категориальные признаки, общие для всех наборов
CAT_COLS = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
# Исходные числовые признаки без инженерии (fnlwgt и education отброшены, см. EDA)
RAW_NUM_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
# Числовые признаки после feature engineering
FE_NUM_COLS = [
    "age", "age_squared", "education_num", "hours_per_week", "experience", "edu_x_hours",
    "log_capital_gain", "log_capital_loss", "has_capital_gain", "has_capital_loss",
    "capital_gain_capped", "overtime", "part_time", "is_married", "is_us",
]

# Непрерывные признаки, для которых строим кусочно-линейное кодирование (бинарным оно не нужно)
FE_PLE_COLS = ["age", "education_num", "hours_per_week", "log_capital_gain", "log_capital_loss", "experience", "edu_x_hours"]


def load_raw(root):
    """Читает сырой CSV; если файла нет, скачивает его с OpenML (data_id: adult, версия 2)."""
    path = Path(root) / "data" / "raw" / "adult.csv"
    if not path.exists():
        from sklearn.datasets import fetch_openml  # импорт здесь: нужен только при первом запуске

        path.parent.mkdir(parents=True, exist_ok=True)
        fetch_openml("adult", version=2, as_frame=True, parser="auto").frame.to_csv(path, index=False)
    return pd.read_csv(path)


def clean(raw):
    """Чистит данные, не меняя исходный файл. Здесь нет ничего, что учится на данных."""
    df = raw.copy()  # сырая таблица остаётся нетронутой
    df.columns = [c.replace("-", "_") for c in df.columns]  # имена без дефисов удобнее в коде
    # Точный дубль с тем же fnlwgt почти наверняка повтор записи. Убираем до разбиения,
    # иначе одна и та же строка может попасть и в train, и в test (утечка)
    df = df.drop_duplicates().reset_index(drop=True)
    # Пропуск в анкете сам несёт информацию, поэтому это отдельная категория, а не мода
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df[TARGET] = (df["class"] == ">50K").astype(int)  # 1 = доход выше 50 тысяч
    return df.drop(columns=["class"])


def add_features(df):
    """Добавляет признаки. Каждый считается по одной строке, поэтому утечки здесь быть не может."""
    out = df.copy()
    out["age_squared"] = out["age"] ** 2  # доход растёт с возрастом, потом падает: нужна парабола
    # Формула Минцера: сколько лет человек мог работать после учёбы
    out["experience"] = (out["age"] - out["education_num"] - 6).clip(lower=0)
    out["edu_x_hours"] = out["education_num"] * out["hours_per_week"]  # образование окупается вместе с часами
    out["log_capital_gain"] = np.log1p(out["capital_gain"])  # сжимаем хвост до 99999
    out["log_capital_loss"] = np.log1p(out["capital_loss"])
    out["has_capital_gain"] = (out["capital_gain"] > 0).astype(int)  # у большинства здесь ноль
    out["has_capital_loss"] = (out["capital_loss"] > 0).astype(int)
    out["capital_gain_capped"] = (out["capital_gain"] == 99999).astype(int)  # 99999 это потолок анкеты
    out["overtime"] = (out["hours_per_week"] > 40).astype(int)  # больше стандартной недели
    out["part_time"] = (out["hours_per_week"] < 35).astype(int)  # неполная занятость
    out["is_married"] = out["marital_status"].isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(int)
    out["is_us"] = (out["native_country"] == "United-States").astype(int)
    return out


def split_data(df, seed=SEED):
    """Делит строки 70/15/15 со stratify, чтобы доля класса совпадала во всех сплитах."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=seed)
    return train, val, test


def split_summary(train, val, test):
    """Таблица для отчёта: размер каждого сплита и доля положительного класса."""
    rows = [("train", train), ("val", val), ("test", test)]
    return pd.DataFrame(
        [{"сплит": n, "строк": len(d), ">50K": int(d[TARGET].sum()), "доля": round(d[TARGET].mean(), 3)} for n, d in rows]
    )
