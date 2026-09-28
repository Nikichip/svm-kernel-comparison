# SVM Kernel Comparison Project — README

## Project Overview
**Title:** SVM Classification with Kernel Comparison (Linear vs Polynomial vs RBF)  
**Course:** Machine Learning (Undergraduate)  
**Tools:** Python 3.11, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

---

## Project Structure

```
svm_kernel_comparison/
├── data_prep.py        # Dataset loading & preprocessing
├── train_models.py     # SVM model training (3 kernels + GridSearchCV)
├── evaluate.py         # Metrics, confusion matrices, summary table
├── visualize.py        # All plot generation functions
├── main.py             # Full pipeline runner (run this!)
├── main.ipynb          # Jupyter Notebook (interactive version)
├── outputs/            # All generated PNG plots saved here
│   ├── decision_boundaries_synthetic.png
│   ├── decision_boundaries_breast_cancer.png
│   ├── gamma_effect_rbf.png
│   ├── metric_comparison_synthetic.png
│   ├── metric_comparison_breast_cancer.png
│   ├── confusion_matrices_synthetic.png
│   ├── confusion_matrices_breast_cancer.png
│   └── gridsearch_heatmap.png
└── README.md
```

---

## How to Run

### Option A — Python script (recommended for quick results)
```bash
cd svm_kernel_comparison
python main.py
```

### Option B — Jupyter Notebook (recommended for the report)
```bash
cd svm_kernel_comparison
jupyter notebook main.ipynb
```
Then: **Kernel → Restart & Run All**

---

## Datasets Used

| Dataset | Type | Samples | Features | Classes |
|---|---|---|---|---|
| `make_moons` (sklearn) | Synthetic | 500 | 2 | 2 (binary) |
| Breast Cancer Wisconsin | Real-world | 569 | 30 | 2 (malignant/benign) |

**Why make_moons?** The two interleaving half-circles are impossible to separate with a straight line — this is the perfect motivating example for why non-linear kernels matter.

---

## Results Summary

### Synthetic Dataset (make_moons, noise=0.25)

| Kernel | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Linear | ~0.83 | ~0.83 | ~0.83 | ~0.83 |
| Polynomial (d=3) | ~0.88 | ~0.88 | ~0.88 | ~0.88 |
| **RBF** | **~0.92** | **~0.92** | **~0.92** | **~0.92** |

> ⚠️ Exact numbers depend on random seed and run. See terminal output for precise values.

### Real Dataset (Breast Cancer Wisconsin)

| Kernel | Accuracy | F1-Score |
|---|---|---|
| Linear | ~0.97 | ~0.97 |
| Polynomial (d=3) | ~0.96 | ~0.96 |
| **RBF** | **~0.98** | **~0.98** |

---

## Key Findings

### 🏆 Best Kernel: RBF (Radial Basis Function)

**Why RBF wins:**
1. **Non-linear flexibility** — The Gaussian kernel can model arbitrarily complex boundaries by mapping data into an infinite-dimensional space.
2. **Gamma control** — The `gamma` parameter lets you tune between smooth (small γ) and tight (large γ) boundaries, giving precise control over bias-variance trade-off.
3. **Fewer hyperparameters than Polynomial** — Only `C` and `gamma` to tune, vs. `C`, `degree`, `coef0` for Poly.

### When Linear Kernel Is Preferred
- When data **is** linearly separable (or close to it).
- When **speed** matters — linear SVM is O(n) training vs. O(n²–n³) for kernel SVMs.
- When feature dimensionality is already very high (e.g., text/NLP with TF-IDF).

### Effect of Gamma (γ) on RBF
| Gamma | Boundary | Risk |
|---|---|---|
| Small (0.01) | Very smooth, wide | Underfitting |
| Medium (1.0) | Balanced | Good generalisation |
| Large (100) | Very complex, wraps training points | Overfitting |

---

## Visualizations Generated

1. **`decision_boundaries_synthetic.png`** — Side-by-side comparison of all 3 kernels on make_moons, with support vectors highlighted.
2. **`decision_boundaries_breast_cancer.png`** — Same for Breast Cancer (2-D slice).
3. **`gamma_effect_rbf.png`** — How small / medium / large γ changes the RBF boundary shape.
4. **`metric_comparison_synthetic.png`** — Grouped bar chart of Accuracy & F1.
5. **`metric_comparison_breast_cancer.png`** — Same for real dataset.
6. **`confusion_matrices_synthetic.png`** — Side-by-side confusion matrices.
7. **`confusion_matrices_breast_cancer.png`** — Same for real dataset.
8. **`gridsearch_heatmap.png`** — C × γ grid showing mean CV accuracy from GridSearchCV.

---

## Dependencies

```
scikit-learn >= 1.2
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
seaborn >= 0.12
notebook (for .ipynb)
```

Install all at once:
```bash
pip install scikit-learn numpy pandas matplotlib seaborn notebook
```

---

## Theoretical Background (Quick Reference)

**What is a Kernel?**  
A kernel function K(x, z) computes the dot product of two data points in a transformed feature space *without* explicitly computing the transformation. This is the **Kernel Trick**.

**Why does Linear SVM fail on non-linear data?**  
Linear SVM finds a hyperplane `w·x + b = 0` in the original feature space. If no such hyperplane separates the classes (e.g., make_moons), the model will have high error regardless of `C`.

**What does RBF achieve?**  
K(x, z) = exp(−γ||x − z||²) measures similarity as a Gaussian decay with distance. Implicitly maps data into an infinite-dimensional Hilbert space where linear separation becomes possible.
