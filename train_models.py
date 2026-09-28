"""
train_models.py
===============
Trains three SVM classifiers with different kernel functions:
  - Linear   : SVM with a straight hyperplane — fast but fails on non-linear data.
  - Polynomial: Maps features into higher-dimensional space using a polynomial.
  - RBF      : Radial Basis Function — the most flexible kernel, uses a Gaussian
               "bump" centred at each support vector.

Also runs GridSearchCV to find the best C & gamma for the RBF kernel,
and trains three RBF models with small / medium / large gamma to illustrate
how gamma controls the decision boundary's smoothness vs. over-fitting.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# ─── Single model trainers ────────────────────────────────────────────────────

def train_linear_svm(X_train, y_train, C=1.0):
    """
    Train an SVM with a LINEAR kernel.

    The linear kernel computes: K(x, z) = x · z
    Best when data is (approximately) linearly separable in the original space.

    Parameters
    ----------
    C : Regularization strength. Large C → low bias, high variance (harder margin).
    """
    model = SVC(kernel='linear', C=C, random_state=42)
    model.fit(X_train, y_train)
    print(f"  [Linear]  Trained — #support vectors: {model.support_vectors_.shape[0]}")
    return model


def train_poly_svm(X_train, y_train, C=1.0, degree=3, coef0=1):
    """
    Train an SVM with a POLYNOMIAL kernel (degree=3 by default).

    Kernel: K(x, z) = (γ x·z + r)^d
    Implicitly maps data into a higher-dimensional feature space without computing
    the mapping explicitly — this is the Kernel Trick.

    Parameters
    ----------
    degree : Polynomial degree. Higher degree → more complex boundary but slower.
    coef0  : Constant term r in the formula above.
    """
    model = SVC(kernel='poly', C=C, degree=degree, coef0=coef0, random_state=42)
    model.fit(X_train, y_train)
    print(f"  [Poly d={degree}] Trained — #support vectors: {model.support_vectors_.shape[0]}")
    return model


def train_rbf_svm(X_train, y_train, C=1.0, gamma='scale'):
    """
    Train an SVM with an RBF (Gaussian) kernel.

    Kernel: K(x, z) = exp(-γ ||x - z||²)
    γ (gamma) controls the 'reach' of each training point:
      - Small γ → wide influence, smoother boundary (may underfit)
      - Large γ → narrow influence, wiggly boundary (may overfit)

    Parameters
    ----------
    gamma : 'scale', 'auto', or a float value.
    """
    model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
    model.fit(X_train, y_train)
    n_sv = model.support_vectors_.shape[0]
    g_val = model.gamma if isinstance(gamma, str) else gamma
    print(f"  [RBF γ={gamma}] Trained — #support vectors: {n_sv}")
    return model


# ─── GridSearchCV for RBF hyperparameter tuning ──────────────────────────────

def tune_rbf_svm(X_train, y_train, cv=5):
    """
    Use GridSearchCV to find the best C and gamma for the RBF kernel.

    Searches over a log-scale grid of C and gamma values.
    Cross-validation score (accuracy) is used to select the winner.

    Returns
    -------
    best_model : Trained SVC with best hyperparameters.
    results_df : pandas DataFrame with all CV results (for optional inspection).
    """
    import pandas as pd

    param_grid = {
        'C':     [0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10]
    }

    grid_search = GridSearchCV(
        SVC(kernel='rbf', random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,      # use all CPU cores
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  [GridSearch] Best params : {grid_search.best_params_}")
    print(f"  [GridSearch] Best CV score: {grid_search.best_score_:.4f}")

    results_df = pd.DataFrame(grid_search.cv_results_)[
        ['param_C', 'param_gamma', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ].sort_values('rank_test_score')

    return grid_search.best_estimator_, results_df


# ─── Gamma effect models ──────────────────────────────────────────────────────

def train_gamma_variants(X_train, y_train, C=10):
    """
    Train three RBF SVMs with different gamma values to illustrate
    how gamma controls the decision boundary smoothness.

    Gamma values chosen to show clear underfitting → good fit → overfitting.

    Returns
    -------
    dict: {'small': model, 'medium': model, 'large': model}
    """
    gamma_values = {
        'small'  : 0.01,   # very smooth boundary → likely underfit
        'medium' : 1.0,    # balanced boundary → usually good fit
        'large'  : 100.0   # very spiky boundary → likely overfit
    }
    models = {}
    print("\n  Training gamma variant models:")
    for label, g in gamma_values.items():
        m = SVC(kernel='rbf', C=C, gamma=g, random_state=42)
        m.fit(X_train, y_train)
        models[label] = m
        print(f"    γ={g:>6} ({label:6s}) — #support vectors: {m.support_vectors_.shape[0]}")
    return models, gamma_values


# ─── Train all models (convenience wrapper used by main) ─────────────────────

def train_all(X_train, y_train):
    """
    Convenience function: trains the three core kernel models and returns them
    as a dict. Used by main.py / main.ipynb.

    Returns
    -------
    dict with keys: 'linear', 'poly', 'rbf'
    """
    print("\n📌 Training SVM models...")
    models = {
        'linear': train_linear_svm(X_train, y_train, C=1.0),
        'poly'  : train_poly_svm(X_train, y_train, C=1.0, degree=3),
        'rbf'   : train_rbf_svm(X_train, y_train, C=1.0, gamma='scale'),
    }
    print("✅ All three kernel models trained.")
    return models


if __name__ == "__main__":
    # Standalone test with a tiny random dataset
    from data_prep import load_synthetic_data
    X_tr, X_te, y_tr, y_te, _ = load_synthetic_data()
    models = train_all(X_tr, y_tr)
    best_rbf, grid_df = tune_rbf_svm(X_tr, y_tr)
    gamma_models, gamma_vals = train_gamma_variants(X_tr, y_tr)
    print("\n✅ train_models.py checks passed.")
