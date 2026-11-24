"""Важность признаков: групповая перестановка и группы коррелированных признаков."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import SEED
from src.metrics import scores

# Признаки датасета: каждое из 10 свойств ядра записано как mean, error и worst
FEATURE_GROUPS = {
    "размер (radius, perimeter, area)": ["radius", "perimeter", "area"],
    "форма контура (concavity, concave points, compactness)": ["concavity", "concave points", "compactness"],
    "текстура (texture)": ["texture"],
    "прочее (smoothness, symmetry, fractal dimension)": ["smoothness", "symmetry", "fractal dimension"],
}


def group_columns(columns):
    """Раскладывает столбцы датасета по смысловым группам."""
    groups = {}
    for group, stems in FEATURE_GROUPS.items():
        # concavity и concave points различаются, поэтому проверяем вхождение точного свойства
        groups[group] = [c for c in columns if any(stem in c for stem in stems)]
    return groups


def group_permutation_importance(model, X, y, groups, n_repeats=30, seed=SEED):
    """Падение ROC-AUC, если перемешать всю группу признаков одной перестановкой строк."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, scores(model, X))  # качество на нетронутых данных
    rows = []
    for name, columns in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))  # одна перестановка на всю группу
            # Внутри группы связи сохраняются, но связь группы с целью разрушена
            shuffled[columns] = X[columns].to_numpy()[order]
            drops.append(base - roc_auc_score(y, scores(model, shuffled)))
        rows.append({"группа": name, "признаков": len(columns), "падение ROC-AUC": np.mean(drops), "std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("падение ROC-AUC", ascending=False, ignore_index=True)
