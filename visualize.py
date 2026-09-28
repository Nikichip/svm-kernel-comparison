"""
visualize.py
============
All plotting functions for the SVM Kernel Comparison project.

Functions:
  - plot_decision_boundaries()  : Side-by-side decision boundaries for 3 kernels.
  - plot_gamma_effect()         : 3 subplots showing small/medium/large gamma RBF.
  - plot_metric_comparison_bar(): Bar chart comparing accuracy & F1 across kernels.

Design choices:
  - Consistent colour palette (tab10) across all plots.
  - Support vectors highlighted with a distinct marker (black star ★).
  - All plots saved to /outputs directory as high-resolution PNGs.
  - plt.close() called after each save to avoid memory leaks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── Colour palette ────────────────────────────────────────────────────────────
CMAP_LIGHT   = 'RdYlBu'   # background mesh colouring
CLASS_COLORS = ['#EF5350', '#42A5F5']  # red / blue for class 0 / 1
SV_COLOR     = 'black'


# ─── Helper: build a 2-D mesh grid over the feature space ──────────────────────

def _make_mesh(X, h=0.04):
    """
    Create a dense grid of points covering the range of X.
    Used to evaluate the model at every point for decision boundary shading.

    h : step size — smaller = finer (but slower) mesh.
    """
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def _plot_single_boundary(ax, model, X_train, y_train, title, h=0.04):
    """
    Draw a filled decision boundary for one model on a given Axes object.
    Also overlays training points and marks support vectors.
    """
    xx, yy = _make_mesh(X_train, h=h)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Shaded decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=CMAP_LIGHT)
    # Decision boundary line
    ax.contour(xx, yy, Z, colors='black', linewidths=0.8, linestyles='--')

    # Scatter training points, coloured by class
    for cls, colour in enumerate(CLASS_COLORS):
        mask = y_train == cls
        ax.scatter(X_train[mask, 0], X_train[mask, 1],
                   c=colour, edgecolors='white', linewidths=0.4,
                   s=35, label=f'Class {cls}', alpha=0.85)

    # Highlight support vectors
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1],
               s=120, facecolors='none', edgecolors=SV_COLOR,
               linewidths=1.5, marker='o', zorder=5, label='Support Vectors')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel("Feature 1 (scaled)", fontsize=9)
    ax.set_ylabel("Feature 2 (scaled)", fontsize=9)
    ax.legend(fontsize=8, loc='lower right')


# ─── Public plotting functions ─────────────────────────────────────────────────

def plot_decision_boundaries(models_dict, X_train, y_train,
                              output_dir="outputs", dataset_label="synthetic",
                              h=0.04):
    """
    Plot decision boundaries for Linear, Polynomial, and RBF kernels side-by-side.

    Works on 2-D data (or 2-D slice of higher-dimensional data).
    Support vectors are highlighted on each subplot.

    Parameters
    ----------
    models_dict : {'linear': model, 'poly': model, 'rbf': model}
    X_train     : Scaled 2-D training features (n_samples × 2).
    y_train     : Training labels.
    output_dir  : Folder to save the PNG.
    dataset_label: String used in title and filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    kernel_labels = {
        'linear': 'Linear Kernel\n(straight hyperplane)',
        'poly'  : 'Polynomial Kernel (d=3)\n(curved boundary)',
        'rbf'   : 'RBF Kernel\n(flexible Gaussian boundary)',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"SVM Decision Boundaries — {dataset_label.title()} Dataset",
        fontsize=16, fontweight='bold', y=1.02
    )

    for ax, (key, model) in zip(axes, models_dict.items()):
        _plot_single_boundary(ax, model, X_train, y_train,
                              title=kernel_labels.get(key, key), h=h)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"decision_boundaries_{dataset_label}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {fname}")
    return fname


def plot_gamma_effect(gamma_models, gamma_values, X_train, y_train,
                      output_dir="outputs", h=0.04):
    """
    Show how varying gamma changes the RBF decision boundary.

    Three subplots: small γ (underfit), medium γ (good fit), large γ (overfit).

    Key insight:
      - Small γ → each support vector influences a wide region → smooth boundary.
      - Large γ → influence is very local → boundary hugs training points → overfit.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Effect of Gamma (γ) on RBF Kernel Decision Boundary",
        fontsize=16, fontweight='bold', y=1.02
    )

    labels = {
        'small' : f"Small γ = {gamma_values['small']}\n(Underfitting — too smooth)",
        'medium': f"Medium γ = {gamma_values['medium']}\n(Good fit — balanced)",
        'large' : f"Large γ = {gamma_values['large']}\n(Overfitting — too complex)",
    }

    for ax, (key, model) in zip(axes, gamma_models.items()):
        _plot_single_boundary(ax, model, X_train, y_train,
                              title=labels[key], h=h)
        n_sv = model.support_vectors_.shape[0]
        ax.text(0.03, 0.97, f"SVs: {n_sv}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fname = os.path.join(output_dir, "gamma_effect_rbf.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {fname}")
    return fname


def plot_metric_comparison_bar(results_df, output_dir="outputs", dataset_label="synthetic"):
    """
    Grouped bar chart comparing Accuracy and F1-Score across all three kernels.

    Clean, report-ready style with value labels on each bar.

    Parameters
    ----------
    results_df : pandas DataFrame with index = kernel names,
                 columns include 'Accuracy' and 'F1-Score'.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics   = ['Accuracy', 'F1-Score']
    kernels   = results_df.index.tolist()
    x         = np.arange(len(kernels))
    width     = 0.30
    colors    = ['#5C85D6', '#E07B54']  # blue for Accuracy, orange for F1

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals   = results_df[metric].values
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=metric,
                        color=color, edgecolor='white', linewidth=0.8)
        # Add value labels on top of each bar
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Kernel Performance Comparison — {dataset_label.title()} Dataset",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"metric_comparison_{dataset_label}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {fname}")
    return fname


def plot_gridsearch_heatmap(grid_results_df, output_dir="outputs"):
    """
    Heatmap of GridSearchCV results: C vs gamma → mean CV accuracy.
    Helps visualize which hyperparameter region works best for RBF.
    """
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    # Pivot: rows = C, cols = gamma
    pivot = grid_results_df.pivot_table(
        index='param_C', columns='param_gamma',
        values='mean_test_score'
    )

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean CV Accuracy'}
    )
    ax.set_title("GridSearchCV: RBF Kernel — C vs Gamma (Mean CV Accuracy)",
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Gamma (γ)", fontsize=10)
    ax.set_ylabel("C (Regularization)", fontsize=10)

    plt.tight_layout()
    fname = os.path.join(output_dir, "gridsearch_heatmap.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {fname}")
    return fname


if __name__ == "__main__":
    print("✅ visualize.py loaded — run main.py or main.ipynb to generate all plots.")
