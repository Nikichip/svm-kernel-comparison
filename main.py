"""
main.py
=======
Entry point for the SVM Kernel Comparison project.
Run this file to execute the full pipeline end-to-end:

    python main.py

What it does:
  1. Loads & preprocesses both datasets (synthetic + real)
  2. Trains Linear, Polynomial, and RBF SVM models
  3. Tunes RBF hyperparameters with GridSearchCV
  4. Trains gamma-variant RBF models (small / medium / large)
  5. Evaluates all models and prints a comparison table
  6. Generates and saves all visualizations to /outputs

All plots are saved as PNGs in the ./outputs/ directory.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')   # suppress sklearn convergence warnings

# Use non-interactive matplotlib backend (no display window needed)
import matplotlib
matplotlib.use('Agg')

# ─── Import project modules ────────────────────────────────────────────────────
from data_prep   import load_synthetic_data, load_real_data
from train_models import train_all, tune_rbf_svm, train_gamma_variants
from evaluate    import evaluate_all_models, print_summary_table, plot_confusion_matrices
from visualize   import (plot_decision_boundaries, plot_gamma_effect,
                         plot_metric_comparison_bar, plot_gridsearch_heatmap)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC DATASET (make_moons)
# ══════════════════════════════════════════════════════════════════════════════

def run_synthetic_pipeline():
    print("\n" + "═"*65)
    print("  SECTION 1 — SYNTHETIC DATASET (make_moons, noise=0.25)")
    print("═"*65)

    # 1.1 Load data
    X_tr, X_te, y_tr, y_te, scaler = load_synthetic_data(
        n_samples=500, noise=0.25, random_state=42
    )

    # 1.2 Train the three kernel models
    models = train_all(X_tr, y_tr)

    # 1.3 Evaluate
    print("\n📊 Evaluation — Synthetic Dataset")
    results, df = evaluate_all_models(models, X_te, y_te, verbose=True)
    print_summary_table(df, dataset_label="Synthetic (make_moons)")

    # 1.4 GridSearchCV on RBF
    print("\n🔍 GridSearchCV — Tuning RBF on Synthetic Data")
    best_rbf, grid_df = tune_rbf_svm(X_tr, y_tr, cv=5)

    # 1.5 Gamma effect
    gamma_models, gamma_vals = train_gamma_variants(X_tr, y_tr, C=10)

    # ── Plots ──
    print("\n🎨 Generating plots for Synthetic Dataset...")

    # Plot 1: Decision boundaries side-by-side
    plot_decision_boundaries(models, X_tr, y_tr,
                             output_dir=OUTPUT_DIR,
                             dataset_label="synthetic", h=0.04)

    # Plot 2: Gamma effect on RBF boundary
    plot_gamma_effect(gamma_models, gamma_vals, X_tr, y_tr,
                      output_dir=OUTPUT_DIR, h=0.04)

    # Plot 3: Metric comparison bar chart
    plot_metric_comparison_bar(df, output_dir=OUTPUT_DIR,
                               dataset_label="synthetic")

    # Plot 4: Confusion matrices
    plot_confusion_matrices(results, output_dir=OUTPUT_DIR,
                            dataset_label="synthetic")

    # Plot 5: GridSearch heatmap
    plot_gridsearch_heatmap(grid_df, output_dir=OUTPUT_DIR)

    return df, best_rbf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REAL DATASET (Breast Cancer Wisconsin)
# ══════════════════════════════════════════════════════════════════════════════

def run_real_pipeline():
    print("\n" + "═"*65)
    print("  SECTION 2 — REAL DATASET (Breast Cancer Wisconsin)")
    print("═"*65)

    # 2.1 Load data (full 30 features for metrics; 2-D slice for plots)
    (X_tr_full, X_te_full, y_tr, y_te, scaler_full,
     X_tr_2d, X_te_2d, scaler_2d, feature_names) = load_real_data(random_state=42)

    # 2.2 Train on full features (for accurate metrics)
    print("\n📌 Training on full 30-feature set (for evaluation)...")
    models_full = train_all(X_tr_full, y_tr)

    # 2.3 Train on 2-D features (for decision boundary plots only)
    print("\n📌 Training on 2-D feature set (for plots)...")
    models_2d = train_all(X_tr_2d, y_tr)

    # 2.4 Evaluate (full features)
    print("\n📊 Evaluation — Breast Cancer Dataset (full features)")
    results, df = evaluate_all_models(models_full, X_te_full, y_te, verbose=True)
    print_summary_table(df, dataset_label="Breast Cancer (Real)")

    # ── Plots ──
    print("\n🎨 Generating plots for Breast Cancer Dataset...")

    # Plot 6: Decision boundaries (2-D slice)
    plot_decision_boundaries(models_2d, X_tr_2d, y_tr,
                             output_dir=OUTPUT_DIR,
                             dataset_label="breast_cancer", h=0.05)

    # Plot 7: Metric comparison bar chart
    plot_metric_comparison_bar(df, output_dir=OUTPUT_DIR,
                               dataset_label="breast_cancer")

    # Plot 8: Confusion matrices
    plot_confusion_matrices(results, output_dir=OUTPUT_DIR,
                            dataset_label="breast_cancer")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*65)
    print("  SVM KERNEL COMPARISON — Linear vs Polynomial vs RBF")
    print("  ML Course Project")
    print("█"*65)

    # Run both pipelines
    df_synthetic, best_rbf = run_synthetic_pipeline()
    df_real = run_real_pipeline()

    # ── Final consolidated summary ──────────────────────────────────────────
    print("\n" + "█"*65)
    print("  FINAL CONSOLIDATED SUMMARY")
    print("█"*65)

    print("\n📋 Synthetic Dataset (make_moons):")
    print(df_synthetic.to_string())

    print("\n📋 Real Dataset (Breast Cancer):")
    print(df_real.to_string())

    print(f"\n✅ All done! Plots saved to: ./{OUTPUT_DIR}/")
    saved = sorted(os.listdir(OUTPUT_DIR))
    for f in saved:
        print(f"   📊 {f}")

    print("\n🏆 Conclusion:")
    best_kernel_syn  = df_synthetic['F1-Score'].idxmax()
    best_kernel_real = df_real['F1-Score'].idxmax()
    print(f"   Synthetic data — Best kernel by F1: {best_kernel_syn}")
    print(f"   Real data      — Best kernel by F1: {best_kernel_real}")
    print(f"\n   RBF kernel (with tuned C & gamma) generally wins on")
    print(f"   non-linear data. Linear kernel is fastest but needs")
    print(f"   linearly separable data to compete.")
