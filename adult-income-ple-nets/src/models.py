"""Baseline-модели."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
SVM_TRAIN_SIZE = 10_000  # SVC растёт примерно квадратично по строкам, поэтому учим на подвыборке


def baseline_models(seed=SEED):
    """Модели без тюнинга: от наивной к сильной."""
    return {
        "Dummy (частый класс)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "kNN (k=25)": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=seed),
        "SVM RBF (10k строк)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def scores_of(model, X):
    """Непрерывная оценка модели: вероятность или расстояние до разделяющей поверхности у SVM."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
