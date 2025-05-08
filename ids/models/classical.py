"""Factory of classical ML models, including an sklearn MLPClassifier for 'mlp'."""
from __future__ import annotations

from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_classical_models() -> Dict[str, object]:
    """Return a dict of model_key -> estimator instances."""
    return {
        "lr": LogisticRegression(max_iter=1000, n_jobs=-1),
        "knn": KNeighborsClassifier(),
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "svm": SVC(kernel="rbf", gamma="scale"),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128,64),
            activation='relu',
            max_iter=300,    
            early_stopping=True,   
            learning_rate_init=1e-3
        )
    }