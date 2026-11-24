"""Предобработка. Всё, что учится на данных, обучается только на train."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MIN_FREQUENCY = 20  # категорию реже 20 строк в train считаем редкой: на ней нельзя надёжно учиться


def make_onehot_preprocessor(num_cols, cat_cols):
    """Scaler для чисел и one-hot для категорий: вход для sklearn-моделей и простых сетей."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                # Редкие и новые категории уходят в общий столбец "infrequent"
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FREQUENCY, sparse_output=False),
                cat_cols,
            ),
        ]
    )


class EmbeddingPreprocessor:
    """Готовит вход для сети с embeddings: числа масштабируются, категории становятся индексами."""

    def __init__(self, num_cols, cat_cols):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=MIN_FREQUENCY
        )

    def fit(self, df):
        """Учит средние, дисперсии и словари категорий. Вызывать только на train."""
        self.scaler.fit(df[self.num_cols])
        self.encoder.fit(df[self.cat_cols])
        return self

    def transform(self, df):
        """Возвращает float-массив чисел и int-массив индексов категорий."""
        x_num = self.scaler.transform(df[self.num_cols]).astype(np.float32)
        # Сдвиг на 1: индекс 0 резервируем для категории, которой не было в train
        x_cat = (self.encoder.transform(df[self.cat_cols]) + 1).astype(np.int64)
        return x_num, x_cat

    def cardinalities(self):
        """Размер словаря каждого embedding с учётом нулевого индекса для новых категорий."""
        sizes = []
        for i, cats in enumerate(self.encoder.categories_):
            infrequent = self.encoder.infrequent_categories_[i]
            n_infrequent = 0 if infrequent is None else len(infrequent)
            # Все редкие категории схлопываются в один индекс
            n = len(cats) - n_infrequent + (1 if n_infrequent else 0)
            sizes.append(n + 1)
        return sizes

    def category_names(self, column):
        """Имена категорий столбца в порядке индексов embedding (нужно для графика)."""
        i = self.cat_cols.index(column)
        infrequent = self.encoder.infrequent_categories_[i]
        rare = set() if infrequent is None else set(infrequent)
        names = ["<new>"] + [c for c in self.encoder.categories_[i] if c not in rare]
        return names + (["<rare>"] if rare else [])


class PiecewiseLinearEncoder:
    """Кусочно-линейное кодирование чисел (PLE, Gorishniy и др., 2022) с границами от дерева.

    Число превращается в столбец значений: 1 для пройденных интервалов, доля для текущего,
    0 для следующих. Так сеть может выучить резкий порог, которого не даёт один столбец.
    """

    def __init__(self, cols, max_bins=16, min_samples_leaf=50, seed=42):
        self.cols = list(cols)
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.edges_ = {}

    def fit(self, df, y):
        """Находит границы по train и его цели: дерево режет там, где меняется доля класса."""
        for col in self.cols:
            tree = DecisionTreeClassifier(
                max_leaf_nodes=self.max_bins, min_samples_leaf=self.min_samples_leaf, random_state=self.seed
            ).fit(df[[col]], y)
            cuts = tree.tree_.threshold[tree.tree_.feature >= 0]  # у листьев feature = -2, их пропускаем
            x = df[col].to_numpy()
            self.edges_[col] = np.unique(np.r_[x.min(), cuts, x.max()])
        return self

    def transform(self, df):
        """Каждому числу сопоставляет заполненность каждого интервала от 0 до 1."""
        parts = []
        for col, edges in self.edges_.items():
            x = df[col].to_numpy(dtype=float)[:, None]
            low, high = edges[:-1], edges[1:]
            parts.append(np.clip((x - low) / (high - low), 0, 1))  # clip для значений вне диапазона train
        return np.hstack(parts).astype(np.float32)
