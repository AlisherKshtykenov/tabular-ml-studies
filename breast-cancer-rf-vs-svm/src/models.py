"""Модели и сетки гиперпараметров."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import SEED

# Сетка для леса: число деревьев, глубина, число признаков на разбиение, размер листа
RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [1, 3],
}

# Сетка для SVM: параметры с приставкой svc__ относятся к шагу "svc" внутри Pipeline
SVM_GRID = {
    "svc__kernel": ["linear", "rbf"],
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1],
}


def make_rf(**params):
    """Random Forest: масштабирование не нужно, деревья сравнивают признак с порогом."""
    # n_jobs=1: SVC работает на одном ядре, поэтому время обучения сравниваем честно
    return RandomForestClassifier(random_state=SEED, n_jobs=1, **params)


def make_svm(scaled=True, **params):
    """SVM в Pipeline: scaler обучается только на тех данных, что переданы в fit."""
    steps = [("svc", SVC(random_state=SEED, **params))]
    if scaled:
        steps.insert(0, ("scaler", StandardScaler()))  # без него SVM зависит от единиц измерения
    return Pipeline(steps)
