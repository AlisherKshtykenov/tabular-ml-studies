"""Feature engineering (создание признаков) и предобработка."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
RAW_NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
ENGINEERED_NUMERIC = [
    "age",
    "day",
    "balance_slog",
    "campaign_log",
    "previous_log",
    "recency",
    "is_negative_balance",
    "was_contacted_before",
    "is_young_or_senior",
    "has_any_loan",
]


def add_features(X):
    """Считает новые признаки по одной строке. Здесь ничего не учится, поэтому утечки нет."""
    X = X.copy()
    # Знаковый логарифм сжимает длинный хвост баланса и сохраняет знак долга
    X["balance_slog"] = np.sign(X["balance"]) * np.log1p(np.abs(X["balance"]))
    X["is_negative_balance"] = (X["balance"] < 0).astype(int)
    # У счётчиков длинный правый хвост (previous доходит до 275), логарифм его сжимает
    X["campaign_log"] = np.log1p(X["campaign"])
    X["previous_log"] = np.log1p(X["previous"])
    # pdays = -1 означает «не звонили». Как число это бессмысленно, поэтому отделяем флагом
    contacted = X["pdays"] != -1
    X["was_contacted_before"] = contacted.astype(int)
    # Свежесть контакта: чем меньше дней прошло, тем ближе к 1. Без контакта строго 0
    X["recency"] = np.where(contacted, 1 / (1 + X["pdays"].clip(lower=0)), 0.0)
    # Зависимость от возраста U-образная, линейная модель сама её не увидит
    X["is_young_or_senior"] = ((X["age"] <= 25) | (X["age"] >= 60)).astype(int)
    X["has_any_loan"] = ((X["housing"] == "yes") | (X["loan"] == "yes")).astype(int)
    return X


def build_preprocessor(feature_set="engineered", extra_numeric=()):
    """Pipeline предобработки. Scaler и encoder обучаются только на том, что передано в fit."""
    numeric = (ENGINEERED_NUMERIC if feature_set == "engineered" else RAW_NUMERIC) + list(extra_numeric)
    columns = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            # handle_unknown="ignore": новая категория в тесте не ломает модель
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    steps = [("columns", columns)]
    if feature_set == "engineered":
        steps.insert(0, ("add_features", FunctionTransformer(add_features)))
    return Pipeline(steps)


def feature_names(preprocessor):
    """Имена столбцов после one-hot кодирования: нужны для коэффициентов логрегрессии."""
    names = preprocessor.named_steps["columns"].get_feature_names_out()
    return pd.Index([name.split("__", 1)[1] for name in names])
