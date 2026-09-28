"""
evaluate.py
===========
Evaluation utilities for trained SVM models.

Functions:
  - evaluate_model()       : Computes accuracy, precision, recall, F1 for one model.
  - evaluate_all_models()  : Runs evaluation on a dict of models, collects results.
  - print_summary_table()  : Pretty-prints a comparison table of all kernels.
  - plot_confusion_matrices(): Saves a side-by-side confusion matrix figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, model_name="Model", verbose=True):
    """
    Evaluate a single trained SVM on the test set.

    Metrics computed:
    - Accuracy  : Overall fraction of correct predictions.
    - Precision : Of all predicted positive, how many are truly positive.
    - Recall    : Of all actual positives, how many did we catch.
    - F1-score  : Harmonic mean of precision and recall — good for imbalanced data.

    Parameters
    ----------
    model      : Trained sklearn SVC instance.
    X_test     : Scaled test features.
    y_test     : True labels for test set.
    model_name : Label to display in output.
    verbose    : Whether to print the classification report.

    Returns
    -------
    dict : {'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'}
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'model_name'      : model_name,
        'accuracy'        : round(acc,  4),
        'precision'       : round(prec, 4),
        'recall'          : round(rec,  4),
        'f1'              : round(f1,   4),
        'confusion_matrix': cm
    }


def evaluate_all_models(models_dict, X_test, y_test, verbose=True):
    """
    Evaluate all models in a dict and collect results into a DataFrame.

    Parameters
    ----------
    models_dict : {'linear': model, 'poly': model, 'rbf': model}

    Returns
    -------
    results : list of dicts (one per model, from evaluate_model)
    df      : pandas DataFrame with summary row per model (no confusion matrix column)
    """
    results = []
    label_map = {
        'linear': 'Linear Kernel',
        'poly'  : 'Polynomial (d=3)',
        'rbf'   : 'RBF Kernel'
    }

    print("\n📊 Evaluating models on test set...")
    for key, model in models_dict.items():
        name = label_map.get(key, key)
        res  = evaluate_model(model, X_test, y_test, model_name=name, verbose=verbose)
        results.append(res)

    # Build a tidy DataFrame (drop the matrix column for display)
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'confusion_matrix'}
        for r in results
    ])
    df = df.rename(columns={
        'model_name': 'Kernel',
        'accuracy'  : 'Accuracy',
        'precision' : 'Precision',
        'recall'    : 'Recall',
        'f1'        : 'F1-Score'
    })
    df = df.set_index('Kernel')
    return results, df


def print_summary_table(df, dataset_label="Synthetic (make_moons)"):
    """
    Pretty-print the final comparison table to stdout.
    Also marks the best value in each metric column with a ★.
    """
    print(f"\n{'='*62}")
    print(f"  FINAL PERFORMANCE SUMMARY — {dataset_label}")
    print(f"{'='*62}")
    print(df.to_string())
    print(f"{'='*62}")

    print("\n  ★ Best per metric:")
    for col in df.columns:
        best_kernel = df[col].idxmax()
        best_val    = df[col].max()
        print(f"    {col:<12}: {best_kernel:<25} ({best_val:.4f})")
    print()


def plot_confusion_matrices(results, output_dir="outputs", dataset_label="synthetic"):
    """
    Plot side-by-side confusion matrices for all three kernels and save as PNG.

    Uses seaborn heatmaps with annotation for clarity.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.suptitle(f"Confusion Matrices — {dataset_label.title()} Dataset",
                 fontsize=15, fontweight='bold', y=1.02)

    for ax, res in zip(axes, results):
        cm = res['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, linecolor='gray',
            ax=ax, cbar=False
        )
        ax.set_title(res['model_name'], fontsize=12, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"confusion_matrices_{dataset_label}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {fname}")
    return fname


if __name__ == "__main__":
    # Standalone test
    from data_prep import load_synthetic_data
    from train_models import train_all

    X_tr, X_te, y_tr, y_te, _ = load_synthetic_data()
    models = train_all(X_tr, y_tr)
    results, df = evaluate_all_models(models, X_te, y_te)
    print_summary_table(df)
    plot_confusion_matrices(results, output_dir="outputs", dataset_label="synthetic")
    print("\n✅ evaluate.py checks passed.")
