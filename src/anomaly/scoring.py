"""One-class feature-space anomaly modelling from notebooks 09 and 10."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class OneClassModels:
    scaler: StandardScaler
    pca: PCA
    mahalanobis: LedoitWolf
    isolation_forest: IsolationForest
    one_class_svm: OneClassSVM | None = None

    def transform(self, X):
        return self.pca.transform(self.scaler.transform(X))

    def score_mahalanobis(self, X):
        Z = self.transform(X)
        diff = Z - self.mahalanobis.location_
        return np.einsum(
            "ij,jk,ik->i",
            diff,
            self.mahalanobis.precision_,
            diff,
        )

    def score_isolation_forest(self, X):
        return -self.isolation_forest.score_samples(self.transform(X))

    def score_ocsvm(self, X):
        if self.one_class_svm is None:
            raise RuntimeError("One-Class SVM was not fitted.")
        return -self.one_class_svm.decision_function(self.transform(X)).ravel()


def fit_one_class_models(
    X_normal,
    *,
    variance_target: float = 0.95,
    max_components: int = 128,
    random_state: int = 42,
    include_ocsvm: bool = True,
) -> OneClassModels:
    """Fit scaler/PCA and detectors using normal/benign training features only."""
    X_normal = np.asarray(X_normal)
    if len(X_normal) < 20:
        raise ValueError("Too few normal training samples for one-class modelling.")

    scaler = StandardScaler().fit(X_normal)
    Z_normal = scaler.transform(X_normal)

    nmax = min(max_components, Z_normal.shape[0] - 1, Z_normal.shape[1])
    if nmax < 2:
        raise ValueError("Too few normal samples for PCA/covariance modelling.")

    probe = PCA(n_components=nmax, svd_solver="full").fit(Z_normal)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    if cumulative[-1] >= variance_target:
        n_components = int(np.searchsorted(cumulative, variance_target) + 1)
    else:
        n_components = int(nmax)

    pca = PCA(n_components=n_components, svd_solver="full").fit(Z_normal)
    Z = pca.transform(Z_normal)

    mahal = LedoitWolf().fit(Z)
    iforest = IsolationForest(
        n_estimators=500,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    ).fit(Z)
    ocsvm = (
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Z)
        if include_ocsvm
        else None
    )
    return OneClassModels(scaler, pca, mahal, iforest, ocsvm)
