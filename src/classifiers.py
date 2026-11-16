"""Independent game-status and conditional-ownership heads.

Independently implemented from the standard probability factorisation
P(H)=(1-P(D))*P(H|alive), also used by FCB-Hacks and the possession literature.
No competitor source or fitted parameters are included.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class TwoHeadClassifier:
    classes_ = np.array([0, 1, 2])

    def __init__(self, **parameters):
        self.parameters = parameters
        self.status = HistGradientBoostingClassifier(**parameters)
        self.ownership = HistGradientBoostingClassifier(**parameters)

    def fit(self, x, y, sample_weight=None):
        y = np.asarray(y)
        if not np.array_equal(np.unique(y), self.classes_):
            raise ValueError("both teams and dead-ball external examples required")
        alive = y != 2
        self.status.fit(x, y == 2, sample_weight=sample_weight)
        weights = None if sample_weight is None else sample_weight[alive]
        self.ownership.fit(x[alive], y[alive], sample_weight=weights)
        return self

    def predict_proba(self, x):
        if not np.array_equal(self.status.classes_, [False, True]) or not np.array_equal(self.ownership.classes_, [0, 1]):
            raise ValueError("invalid learned head class ordering")
        dead = self.status.predict_proba(x)[:, 1]
        team = self.ownership.predict_proba(x)
        return np.c_[(1-dead)*team[:, 0], (1-dead)*team[:, 1], dead]
