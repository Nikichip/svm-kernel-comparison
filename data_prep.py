"""
data_prep.py
============
Handles loading and preprocessing of both datasets used in this project:
  1. Synthetic dataset (make_moons) — non-linearly separable, ideal for showing
     why kernel choice matters.
  2. Real-world dataset (Breast Cancer) — binary classification from sklearn.

Preprocessing steps:
  - Train/test split (80/20, stratified for class balance)
  - Feature standardization using StandardScaler (zero mean, unit variance)
    This is crucial for SVM since it uses distance-based computations.
"""

import numpy as np
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_synthetic_data(n_samples=500, noise=0.25, random_state=42):
    """
    Generate a non-linearly separable synthetic dataset using make_moons.

    Why make_moons?
      - Two interleaving half-circles → a linear boundary simply cannot separate them.
      - This clearly motivates the need for non-linear kernels (Poly, RBF).

    Parameters
    ----------
    n_samples   : int   — Number of data points to generate.
    noise       : float — Gaussian noise added to data (higher = harder problem).
    random_state: int   — Seed for reproducibility.

    Returns
    -------
    X_train_sc, X_test_sc : Scaled feature arrays (numpy ndarray)
    y_train, y_test       : Label arrays (0 or 1)
    scaler                : Fitted StandardScaler (needed for decision boundary plots)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # 80/20 split, stratified so both classes are equally represented in each set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Standardize: fit on train only, transform both — avoids data leakage
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[Synthetic] Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")
    print(f"[Synthetic] Class distribution — Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler


def load_real_data(random_state=42):
    """
    Load the Breast Cancer Wisconsin dataset (binary classification).

    30 features describing cell nuclei characteristics, target = malignant/benign.
    We use only the first 2 features for 2-D visualisation; all features for metrics.

    Returns
    -------
    Two tuples:
      (X_train_sc, X_test_sc, y_train, y_test, scaler)  — full 30-feature version
      (X_train_2d, X_test_2d, y_train, y_test, scaler2d) — 2-feature version for plots
    Also returns feature_names for reference.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Full feature set for evaluation metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 2-D slice (features 0 & 1) for decision boundary plots
    X_2d = X[:, :2]
    X_train_2d_raw, X_test_2d_raw, _, _ = train_test_split(
        X_2d, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler2d = StandardScaler()
    X_train_2d = scaler2d.fit_transform(X_train_2d_raw)
    X_test_2d  = scaler2d.transform(X_test_2d_raw)

    print(f"\n[Real Data] Dataset: Breast Cancer Wisconsin")
    print(f"[Real Data] Full features — Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")
    print(f"[Real Data] 2D features  — Train: {X_train_2d.shape}, Test: {X_test_2d.shape}")
    print(f"[Real Data] Class distribution — Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    return (X_train_sc, X_test_sc, y_train, y_test, scaler,
            X_train_2d, X_test_2d, scaler2d, feature_names)


if __name__ == "__main__":
    # Quick sanity check — run this file standalone to verify data loading
    load_synthetic_data()
    load_real_data()
    print("\n✅ Data preparation complete.")
