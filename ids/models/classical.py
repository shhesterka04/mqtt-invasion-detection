"""Factory of scikit‑learn models (classic ML + MLP)."""
from __future__ import annotations
from typing import Dict

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def get_classical_models() -> Dict[str, object]:
    """Return dict model_key → estimator instance."""
    return {
        # базовые
        "dt":  DecisionTreeClassifier(class_weight="balanced", random_state=0),
        "svm_lin": LinearSVC(C=1.0, class_weight="balanced"),      
        "svm_rbf": SVC(kernel="rbf", gamma="scale", class_weight="balanced"),
        "lr":  LogisticRegression(max_iter=1000, n_jobs=-1,
                                  class_weight="balanced"),
        "nb":  GaussianNB(),                                  
        "knn": KNeighborsClassifier(n_neighbors=5),
        "rf":  RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                     class_weight="balanced_subsample",
                                     random_state=0),
        "xgb": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=1.0, n_jobs=-1, random_state=0,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            random_state=0,
        ),
    }
