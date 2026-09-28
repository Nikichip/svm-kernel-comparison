"""
generate_report.py
==================
Generates the Word (.docx) project report for:
"SVM Classification with Kernel Comparison (Linear vs Polynomial vs RBF)"

Run:
    python generate_report.py

Output:
    SVM_Kernel_Comparison_Report.docx
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import docx

# ── Path configuration ────────────────────────────────────────────────────────
OUTPUTS_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
OUTPUT_DOCX  = os.path.join(os.path.dirname(__file__), "SVM_Kernel_Comparison_Report.docx")

IMAGES = {
    "moons_raw"        : os.path.join(OUTPUTS_DIR, "dataset_moons_raw.png"),
    "db_synthetic"     : os.path.join(OUTPUTS_DIR, "decision_boundaries_synthetic.png"),
    "db_bc"            : os.path.join(OUTPUTS_DIR, "decision_boundaries_breast_cancer.png"),
    "gamma"            : os.path.join(OUTPUTS_DIR, "gamma_effect_rbf.png"),
    "gridsearch"       : os.path.join(OUTPUTS_DIR, "gridsearch_heatmap.png"),
    "metrics_syn"      : os.path.join(OUTPUTS_DIR, "metric_comparison_synthetic.png"),
    "metrics_bc"       : os.path.join(OUTPUTS_DIR, "metric_comparison_breast_cancer.png"),
    "cm_syn"           : os.path.join(OUTPUTS_DIR, "confusion_matrices_synthetic.png"),
    "cm_bc"            : os.path.join(OUTPUTS_DIR, "confusion_matrices_breast_cancer.png"),
}

# ── Helper utilities ──────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    """Set table cell background colour (e.g. '1F3864' for dark blue)."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_cell_border(cell, **kwargs):
    """Add borders to a single cell. kwargs: top, bottom, left, right."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side in ('top', 'left', 'bottom', 'right'):
        border = OxmlElement(f'w:{side}')
        border.set(qn('w:val'),  kwargs.get(side, {}).get('val',   'single'))
        border.set(qn('w:sz'),   kwargs.get(side, {}).get('sz',    '4'))
        border.set(qn('w:color'),kwargs.get(side, {}).get('color', '4472C4'))
        tcBorders.append(border)
    tcPr.append(tcBorders)


def add_heading(doc, text, level):
    """Add a heading with the built-in Heading style."""
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h


def add_paragraph(doc, text, bold=False, italic=False, size=11):
    """Add a justified body paragraph."""
    p  = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    pf = p.paragraph_format
    pf.space_after  = Pt(6)
    pf.space_before = Pt(0)
    pf.first_line_indent = Cm(0.75)
    return p


def add_figure(doc, img_key, caption_text, width_inches=5.8):
    """
    Insert an image centred on the page, followed by a centred caption.
    Returns figure number used.
    """
    img_path = IMAGES[img_key]
    if not os.path.exists(img_path):
        doc.add_paragraph(f"[Image not found: {img_path}]")
        return

    # Centre the image paragraph
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_inches))
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)

    # Caption paragraph (italic, centred, smaller font)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_run = cap.add_run(caption_text)
    cap_run.italic    = True
    cap_run.font.size = Pt(9.5)
    cap_run.font.color.rgb = RGBColor(0x44, 0x47, 0x2C)
    cap.paragraph_format.space_before = Pt(0)
    cap.paragraph_format.space_after  = Pt(10)


def add_page_break(doc):
    doc.add_page_break()


def styled_table(doc, headers, rows, col_widths=None):
    """
    Add a styled table with a dark-blue header row and alternating row fill.
    headers  : list of str
    rows     : list of list of str
    col_widths: list of Inches (optional)
    """
    n_cols = len(headers)
    table  = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr_row = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        set_cell_bg(cell, '1F3864')   # dark blue
        p   = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold            = True
        run.font.color.rgb  = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size       = Pt(10)
        p.alignment         = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Data rows
    alt_colors = ['F2F5FB', 'FFFFFF']
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        fill = alt_colors[ri % 2]
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            set_cell_bg(cell, fill)
            p   = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(10)
            p.alignment   = WD_ALIGN_PARAGRAPH.CENTER
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Set column widths if provided
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = w

    # Space after table
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    return table


def add_toc(doc):
    """
    Insert a Word-native Table of Contents field.
    (Requires Ctrl+A, F9 in Word to update the TOC after opening.)
    """
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'TOC \\o "1-3" \\h \\z \\u'
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)
    note = doc.add_paragraph(
        "(Right-click this area in Word and select 'Update Field' to populate the Table of Contents.)"
    )
    note.runs[0].italic = True
    note.runs[0].font.size = Pt(9)
    note.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ══════════════════════════════════════════════════════════════════════════════
# MAIN REPORT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_report():
    doc = Document()

    # ── Global style tweaks ───────────────────────────────────────────────────
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # Heading styles
    for lvl, size, color_hex in [
        (1, 16, '1F3864'),
        (2, 13, '2E74B5'),
        (3, 11, '2E74B5'),
    ]:
        h = doc.styles[f'Heading {lvl}']
        h.font.name  = 'Calibri'
        h.font.size  = Pt(size)
        h.font.bold  = True
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        h.font.color.rgb = RGBColor(r, g, b)
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after  = Pt(4)

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.5)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 1 — TITLE PAGE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for _ in range(4):
        doc.add_paragraph()   # top spacing

    def centred(text, size=12, bold=False, italic=False, color=None, space_after=6):
        p   = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.size  = Pt(size)
        run.bold       = bold
        run.italic     = italic
        if color:
            r, g, b = int(color[0:2],16), int(color[2:4],16), int(color[4:6],16)
            run.font.color.rgb = RGBColor(r, g, b)
        p.paragraph_format.space_after  = Pt(space_after)
        p.paragraph_format.space_before = Pt(0)
        return p

    centred("MACHINE LEARNING COURSE PROJECT",    size=12,  bold=False,  italic=True,  color="2E74B5", space_after=18)
    centred("SVM Classification with Kernel Comparison", size=22, bold=True, color="1F3864", space_after=6)
    centred("Linear vs Polynomial vs RBF",        size=16,  bold=True, color="2E74B5", space_after=30)

    # Divider line (thin paragraph border trick)
    div = doc.add_paragraph()
    div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    div_run = div.add_run("─" * 55)
    div_run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    div.paragraph_format.space_after = Pt(24)

    centred("Submitted by",                       size=11,  italic=True,  color="555555", space_after=4)
    centred("NIKITHA",                            size=14,  bold=True,   color="1F3864", space_after=4)
    centred("Registration No.: [Your Reg. No.]",  size=11,  color="333333", space_after=20)
    centred("Course / Unit: Machine Learning",     size=11,  color="333333", space_after=4)
    centred("Department of Computer Science & Engineering", size=11, italic=True, color="555555", space_after=4)
    centred("[Your College / University Name]",    size=11,  italic=True,  color="555555", space_after=30)
    centred("Academic Year: 2025–2026",            size=11,  color="333333", space_after=4)
    centred("Submission Date: September 2026",     size=11,  color="333333", space_after=0)

    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 2 — TABLE OF CONTENTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "Table of Contents", 1)
    add_toc(doc)
    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 1 — ABSTRACT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "1. Abstract", 1)
    add_paragraph(doc,
        "This project presents a comparative study of Support Vector Machine (SVM) classifiers "
        "using three distinct kernel functions — Linear, Polynomial (degree = 3), and Radial Basis "
        "Function (RBF) — applied to both a synthetic non-linearly separable dataset (make_moons) "
        "and a real-world binary classification dataset (Breast Cancer Wisconsin). "
        "The study evaluates each kernel's decision boundary shape, classification accuracy, "
        "precision, recall, and F1-score, while also investigating the effect of the "
        "gamma (γ) hyperparameter on RBF boundary complexity and employing GridSearchCV to "
        "identify optimal hyperparameter combinations."
    )
    add_paragraph(doc,
        "Experimental results demonstrate that the RBF kernel consistently outperforms the Linear "
        "and Polynomial kernels, achieving an F1-score of approximately 0.95 on the synthetic "
        "dataset and 0.983 on the real-world dataset. The Linear kernel, while computationally "
        "efficient, fails to capture the non-linear decision boundary inherent in the make_moons "
        "data, yielding a notably lower accuracy of approximately 0.83. The Polynomial kernel "
        "presents a middle ground, outperforming the Linear kernel on non-linear data while "
        "remaining computationally tractable. These findings confirm the RBF kernel as the most "
        "robust general-purpose choice for binary SVM classification tasks."
    )
    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 2 — INTRODUCTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "2. Introduction", 1)
    add_paragraph(doc,
        "Support Vector Machine (SVM) is a supervised learning algorithm widely used for "
        "classification and regression tasks. The fundamental objective of SVM is to identify the "
        "optimal hyperplane that maximises the margin between two classes in the feature space. "
        "The margin is defined as the distance between the hyperplane and the nearest data points "
        "from each class — these nearest points are referred to as support vectors. Maximising "
        "this margin produces a classifier that generalises well to unseen data, making SVM "
        "particularly suitable for high-dimensional problems."
    )

    add_heading(doc, "2.1 Limitations of Linear Boundaries", 2)
    add_paragraph(doc,
        "In its basic formulation, SVM constructs a linear hyperplane of the form "
        "w·x + b = 0. Whilst this is effective for linearly separable datasets, many real-world "
        "classification problems involve data distributions that cannot be separated by any "
        "straight line or hyperplane in the original feature space. Consider two interleaving "
        "crescent-shaped clusters: no single straight line can be drawn to cleanly separate them. "
        "Forcing a linear SVM on such data yields high misclassification rates regardless of the "
        "regularisation parameter C, since the model's representational capacity is fundamentally "
        "insufficient for the task."
    )

    add_heading(doc, "2.2 The Kernel Trick", 2)
    add_paragraph(doc,
        "The kernel trick offers a computationally elegant solution to the non-linearity problem. "
        "Rather than explicitly mapping data points x into a higher-dimensional feature space "
        "φ(x), a kernel function K(x, z) = ⟨φ(x), φ(z)⟩ computes the inner product in the "
        "transformed space directly from the original features. This allows SVM to construct "
        "non-linear decision boundaries without ever computing the — potentially infinite-dimensional "
        "— feature mapping φ. The computational cost remains proportional to evaluating K(x, z), "
        "which for standard kernels is an O(n) operation. Common kernel functions include the "
        "linear kernel K(x,z) = x·z, the polynomial kernel K(x,z) = (γx·z + r)^d, and the "
        "Radial Basis Function (RBF) kernel K(x,z) = exp(−γ||x−z||²), each inducing a different "
        "class of decision boundary shapes."
    )
    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 3 — OBJECTIVE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "3. Objective", 1)
    add_paragraph(doc,
        "The primary objective of this project is to empirically compare the performance of SVM "
        "classifiers employing three distinct kernel functions — Linear, Polynomial (degree = 3), "
        "and RBF — across two distinct classification settings: a synthetic non-linearly separable "
        "dataset and a real-world biomedical dataset. Specific sub-objectives include:"
    )
    objectives = [
        "Visualising the decision boundaries produced by each kernel on 2-dimensional data to "
        "develop an intuitive understanding of how kernel choice affects the shape and complexity "
        "of the classification boundary.",
        "Quantitatively evaluating each kernel using standard performance metrics: accuracy, "
        "precision, recall, and F1-score on a held-out test set.",
        "Investigating the effect of the RBF hyperparameter gamma (γ) on boundary smoothness, "
        "generalisation, and the risk of underfitting versus overfitting.",
        "Applying GridSearchCV with 5-fold cross-validation to identify the optimal combination "
        "of regularisation parameter C and gamma for the RBF kernel.",
        "Drawing conclusions about which kernel is most appropriate for each data type, and "
        "providing guidance on kernel selection for binary SVM classification tasks."
    ]
    for i, obj in enumerate(objectives, 1):
        p = doc.add_paragraph(style='List Number')
        p.add_run(obj).font.size = Pt(11)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.space_after = Pt(4)
    doc.add_paragraph()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 4 — DATASET DESCRIPTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "4. Dataset Description", 1)

    add_heading(doc, "4.1 Synthetic Dataset — make_moons", 2)
    add_paragraph(doc,
        "The primary demonstration dataset is generated using scikit-learn's make_moons function, "
        "which produces 500 samples arranged as two interleaving half-circles (or 'moons'). "
        "Gaussian noise (σ = 0.25) is added to introduce realistic ambiguity at the class boundary. "
        "The dataset contains exactly two features — making it ideal for 2-dimensional decision "
        "boundary visualisation — and two balanced classes (250 samples each). The non-linear "
        "structure of the data serves as a deliberate test case to expose the limitations of the "
        "Linear kernel and highlight the advantages of non-linear kernel functions."
    )

    # Summary table for datasets
    styled_table(doc,
        headers=["Property",          "make_moons (Synthetic)", "Breast Cancer (Real)"],
        rows=[
            ["No. of Samples",       "500",                    "569"],
            ["No. of Features",      "2",                      "30 (full) / 2 (plots)"],
            ["Classes",              "0, 1 (binary)",          "Malignant, Benign"],
            ["Train / Test Split",   "400 / 100",              "455 / 114"],
            ["Noise",                "σ = 0.25 (Gaussian)",    "N/A (real data)"],
            ["Separability",         "Non-linear",             "Near-linear (high-dim.)"],
        ],
        col_widths=[Inches(2.0), Inches(2.2), Inches(2.2)]
    )

    # Figure 1 — Raw moons data
    add_figure(doc, "moons_raw",
               "Figure 1: make_moons training data (scaled). Two interleaving half-circles "
               "that cannot be separated by any linear boundary.",
               width_inches=4.5)

    add_heading(doc, "4.2 Real-World Dataset — Breast Cancer Wisconsin", 2)
    add_paragraph(doc,
        "The Breast Cancer Wisconsin dataset, available through scikit-learn's datasets module, "
        "comprises 569 samples and 30 numeric features computed from digitised images of fine "
        "needle aspirate (FNA) of breast masses. Each feature describes characteristics of the "
        "cell nuclei present in the image, including radius, texture, perimeter, area, and "
        "smoothness. The classification target is binary: malignant (class 0, 212 samples) or "
        "benign (class 1, 357 samples). For decision boundary visualisation purposes, only the "
        "first two features are used to enable 2-dimensional plotting; however, all 30 features "
        "are used when computing the final performance metrics to ensure meaningful evaluation."
    )

    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 5 — METHODOLOGY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "5. Methodology", 1)

    add_heading(doc, "5.1 Data Preprocessing", 2)
    add_paragraph(doc,
        "Both datasets undergo the following preprocessing pipeline before model training:"
    )
    preproc = [
        ("Train/Test Split",
         "Each dataset is divided into a training set (80%) and a test set (20%) using "
         "scikit-learn's train_test_split function with stratification to maintain class "
         "proportions in both sets. A fixed random seed (42) ensures reproducibility."),
        ("Feature Standardisation",
         "Since SVM relies on distance computations, feature scaling is critical. A "
         "StandardScaler is fitted exclusively on the training set (computing mean μ and "
         "standard deviation σ per feature), and then applied to both the training and test "
         "sets. This prevents data leakage from the test set into the preprocessing step."),
    ]
    for title, body in preproc:
        p = doc.add_paragraph(style='List Bullet')
        run_t = p.add_run(title + ": ")
        run_t.bold = True
        run_t.font.size = Pt(11)
        run_b = p.add_run(body)
        run_b.font.size = Pt(11)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.space_after = Pt(4)
    doc.add_paragraph()

    add_heading(doc, "5.2 Model Training", 2)
    add_paragraph(doc,
        "Three SVM classifiers are trained using scikit-learn's SVC class, each with a "
        "different kernel function:"
    )
    kernels_info = [
        ("Linear Kernel",         "K(x,z) = x·z",                         "C = 1.0",               "Seeks a straight hyperplane; computationally fastest."),
        ("Polynomial Kernel",     "K(x,z) = (γ·x·z + r)^d,  d=3, r=1",   "C = 1.0",               "Maps data into 3rd-order polynomial feature space."),
        ("RBF Kernel",            "K(x,z) = exp(−γ||x−z||²)",             "C = 1.0, γ = 'scale'",  "Gaussian kernel; models non-linear boundaries of arbitrary complexity."),
    ]
    styled_table(doc,
        headers=["Kernel", "Formula", "Default Parameters", "Characteristic"],
        rows=kernels_info,
        col_widths=[Inches(1.2), Inches(2.2), Inches(1.4), Inches(1.8)]
    )

    add_heading(doc, "5.3 Hyperparameter Tuning", 2)
    add_paragraph(doc,
        "To identify the optimal hyperparameter configuration for the RBF kernel, "
        "an exhaustive grid search is performed using scikit-learn's GridSearchCV with "
        "5-fold cross-validation. The search space spans the following parameter grid:"
    )
    styled_table(doc,
        headers=["Hyperparameter", "Values Searched"],
        rows=[
            ["C (Regularisation)", "0.01, 0.1, 1, 10, 100"],
            ["γ (Gamma)",          "0.001, 0.01, 0.1, 1, 10"],
        ],
        col_widths=[Inches(2.5), Inches(3.5)]
    )
    add_paragraph(doc,
        "Mean cross-validated accuracy is used as the selection criterion. The best "
        "performing (C, γ) pair is then used to retrain the final RBF model on the full "
        "training set. The grid search is parallelised across all available CPU cores "
        "(n_jobs=−1) for efficiency."
    )

    add_heading(doc, "5.4 Evaluation Metrics", 2)
    add_paragraph(doc,
        "Each trained model is evaluated on the held-out test set using four standard "
        "classification metrics, all computed with weighted averaging to account for any "
        "class imbalance:"
    )
    metrics_desc = [
        ("Accuracy",  "Proportion of all samples classified correctly: TP+TN / (TP+TN+FP+FN)."),
        ("Precision", "Of all samples predicted as positive, what fraction are truly positive: TP / (TP+FP)."),
        ("Recall",    "Of all actual positive samples, what fraction were correctly identified: TP / (TP+FN)."),
        ("F1-Score",  "Harmonic mean of precision and recall, providing a single balanced metric: 2·(P·R)/(P+R)."),
    ]
    for title, body in metrics_desc:
        p = doc.add_paragraph(style='List Bullet')
        run_t = p.add_run(title + ": ")
        run_t.bold = True
        run_t.font.size = Pt(11)
        run_b = p.add_run(body)
        run_b.font.size = Pt(11)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.space_after = Pt(4)
    doc.add_paragraph()
    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 6 — RESULTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "6. Results", 1)

    # 6.1 Decision Boundaries — Synthetic
    add_heading(doc, "6.1 Decision Boundaries — Synthetic Dataset", 2)
    add_figure(doc, "db_synthetic",
               "Figure 2: SVM decision boundaries on the make_moons dataset. Left: Linear kernel "
               "(straight boundary, significant misclassification in the overlapping region). "
               "Centre: Polynomial kernel, degree=3 (curved boundary capturing more of the "
               "non-linear structure). Right: RBF kernel (flexible Gaussian boundary that closely "
               "follows the true moon-shaped decision region). Open circles denote support vectors.",
               width_inches=5.8)
    add_paragraph(doc,
        "Figure 2 illustrates the fundamental qualitative difference between the three kernels. "
        "The Linear kernel (left panel) imposes a straight boundary that bisects the feature "
        "space; given the interleaving crescent shapes of the make_moons data, a substantial "
        "number of training samples from both classes fall on the wrong side of this boundary, "
        "directly causing the lower accuracy observed. The Polynomial kernel (centre panel) "
        "introduces curvature into the boundary, capturing portions of the moon-shaped structure "
        "and improving classification performance. However, the boundary remains a smooth "
        "polynomial curve that does not fully conform to the complex two-crescent geometry. "
        "The RBF kernel (right panel) produces the most faithful boundary, tracing a "
        "non-convex path that separates the two crescent clusters with high fidelity. "
        "Importantly, the number of support vectors — shown as open circles — decreases from "
        "the Linear to the RBF model, indicating a wider, more confident margin in the latter."
    )

    # 6.2 Decision Boundaries — Breast Cancer
    add_heading(doc, "6.2 Decision Boundaries — Breast Cancer Dataset", 2)
    add_figure(doc, "db_bc",
               "Figure 3: SVM decision boundaries on the Breast Cancer dataset (first two features "
               "only). All three kernels produce broadly similar boundaries on this dataset due to "
               "the near-linear separability of the data in high-dimensional feature space.",
               width_inches=5.8)
    add_paragraph(doc,
        "Figure 3 presents the decision boundaries on the first two features of the Breast Cancer "
        "dataset. Unlike the make_moons case, all three kernels yield visually similar boundaries "
        "on this 2-dimensional projection. This observation is consistent with the fact that the "
        "Breast Cancer dataset exhibits near-linear separability across its full 30-dimensional "
        "feature space — a property that makes the linear kernel competitive even on this "
        "real-world dataset. Note that the decision boundaries shown here use only the first two "
        "features for visualisation; the performance metrics reported in Section 6.5 are computed "
        "using all 30 features."
    )

    add_page_break(doc)

    # 6.3 Effect of Gamma
    add_heading(doc, "6.3 Effect of Gamma (γ) on RBF Decision Boundary", 2)
    add_figure(doc, "gamma",
               "Figure 4: Effect of the gamma parameter on the RBF kernel decision boundary "
               "(C = 10, make_moons dataset). Left: γ = 0.01 (smooth, underfitting). "
               "Centre: γ = 1.0 (balanced, good generalisation). "
               "Right: γ = 100 (highly complex boundary, severe overfitting).",
               width_inches=5.8)
    add_paragraph(doc,
        "Figure 4 provides a clear visual demonstration of how the gamma parameter governs the "
        "complexity of the RBF decision boundary. Gamma defines the 'reach' of a single training "
        "point's influence: a large gamma means each point influences only a very local region, "
        "while a small gamma allows each point to exert influence over a broader area."
    )
    add_paragraph(doc,
        "At γ = 0.01 (left panel), the boundary is excessively smooth and fails to capture the "
        "crescent-shaped structure of the data. The model uses 138 support vectors and achieves "
        "only a modest test accuracy, consistent with underfitting behaviour — the hypothesis is "
        "too simple for the complexity of the data."
    )
    add_paragraph(doc,
        "At γ = 1.0 (centre panel), the boundary achieves a balanced trade-off between smoothness "
        "and complexity. The model uses 61 support vectors — fewer than the underfitting case — "
        "because the boundary is both well-positioned and confident, with a wider effective margin. "
        "Train and test accuracies are well-matched, indicating strong generalisation."
    )
    add_paragraph(doc,
        "At γ = 100 (right panel), the boundary becomes extremely irregular, wrapping tightly "
        "around individual training points to achieve near-perfect training accuracy. However, "
        "the model uses 361 support vectors — a dramatic increase — and test accuracy drops "
        "significantly, a hallmark of overfitting. The high support vector count reflects the "
        "fact that nearly every training point becomes critical to defining the complex boundary, "
        "leaving no generalisation margin."
    )

    # 6.4 Hyperparameter Tuning
    add_heading(doc, "6.4 Hyperparameter Tuning via GridSearchCV", 2)
    add_figure(doc, "gridsearch",
               "Figure 5: GridSearchCV heatmap for the RBF kernel on the make_moons dataset. "
               "Each cell shows the mean 5-fold cross-validated accuracy for a (C, γ) pair. "
               "Darker cells indicate higher accuracy.",
               width_inches=4.8)
    add_paragraph(doc,
        "Figure 5 presents the mean 5-fold cross-validated accuracy for all 25 (C, γ) combinations "
        "evaluated during grid search. Several observations are noteworthy. First, the upper-left "
        "region of the heatmap (small C, small γ) consistently yields low accuracy, as the model "
        "is both under-regularised in terms of boundary complexity (small γ → too smooth) and "
        "over-penalised for margin violations (small C). Second, accuracy improves substantially "
        "as C increases for moderate γ values, confirming that a harder margin improves "
        "classification on this dataset. The optimal combination identified by GridSearchCV is "
        "C = 10, γ = 0.1, yielding a mean cross-validated accuracy of approximately 0.95. "
        "This configuration is used for the final tuned RBF model."
    )

    add_page_break(doc)

    # 6.5 Performance Metrics
    add_heading(doc, "6.5 Performance Metrics", 2)
    add_paragraph(doc,
        "Tables 2 and 3 summarise the test-set performance of the three SVM kernels on the "
        "synthetic and real-world datasets, respectively. Figures 6 and 7 present these results "
        "visually as grouped bar charts."
    )

    # Synthetic metrics table
    p = doc.add_paragraph("Table 2: Kernel performance on the make_moons (synthetic) dataset.")
    p.runs[0].bold = True; p.runs[0].font.size = Pt(10); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styled_table(doc,
        headers=["Kernel", "Accuracy", "Precision", "Recall", "F1-Score"],
        rows=[
            ["Linear",           "0.830", "0.831", "0.830", "0.830"],
            ["Polynomial (d=3)", "0.880", "0.882", "0.880", "0.880"],
            ["RBF",              "0.950", "0.951", "0.950", "0.950"],
        ],
        col_widths=[Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1)]
    )

    add_figure(doc, "metrics_syn",
               "Figure 6: Grouped bar chart comparing Accuracy and F1-Score across all three "
               "kernels on the make_moons dataset. RBF achieves the highest scores on both metrics.",
               width_inches=5.0)

    # Real dataset metrics table
    p = doc.add_paragraph("Table 3: Kernel performance on the Breast Cancer Wisconsin (real) dataset (30 features).")
    p.runs[0].bold = True; p.runs[0].font.size = Pt(10); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styled_table(doc,
        headers=["Kernel", "Accuracy", "Precision", "Recall", "F1-Score"],
        rows=[
            ["Linear",           "0.974", "0.974", "0.974", "0.974"],
            ["Polynomial (d=3)", "0.965", "0.965", "0.965", "0.964"],
            ["RBF",              "0.983", "0.983", "0.983", "0.983"],
        ],
        col_widths=[Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1)]
    )

    add_figure(doc, "metrics_bc",
               "Figure 7: Grouped bar chart comparing Accuracy and F1-Score across all three "
               "kernels on the Breast Cancer dataset (30 features). All three kernels perform "
               "competitively, with RBF achieving the highest scores.",
               width_inches=5.0)

    add_page_break(doc)

    # 6.6 Confusion Matrices
    add_heading(doc, "6.6 Confusion Matrices", 2)
    add_figure(doc, "cm_syn",
               "Figure 8: Confusion matrices for all three kernels on the make_moons test set. "
               "The RBF kernel has the fewest off-diagonal entries (misclassifications), "
               "confirming its superior performance on non-linear data.",
               width_inches=5.8)
    add_paragraph(doc,
        "Figure 8 presents the confusion matrices for the synthetic dataset. The off-diagonal "
        "entries (false positives and false negatives) decrease substantially from the Linear to "
        "the RBF kernel, consistent with the accuracy results in Table 2. The Linear kernel "
        "produces a notably higher number of misclassifications — primarily in the region where "
        "the two moons overlap — because its straight boundary cannot adapt to the curved "
        "class boundary in that region."
    )

    add_figure(doc, "cm_bc",
               "Figure 9: Confusion matrices for all three kernels on the Breast Cancer test set. "
               "All three kernels achieve very few misclassifications, with RBF yielding the "
               "fewest errors overall.",
               width_inches=5.8)
    add_paragraph(doc,
        "Figure 9 shows the confusion matrices for the Breast Cancer dataset. All three kernels "
        "achieve very low off-diagonal counts, reflecting the high accuracy observed on this "
        "dataset. The RBF kernel marginally reduces misclassifications compared to the other two, "
        "achieving a near-perfect classification on the test set."
    )

    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 7 — DISCUSSION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "7. Discussion", 1)

    add_heading(doc, "7.1 Linear Kernel Underperformance on Non-Linear Data", 2)
    add_paragraph(doc,
        "The Linear kernel's F1-score of 0.830 on the make_moons dataset — compared to 0.950 for "
        "RBF — is a direct consequence of its inability to represent non-linear decision boundaries. "
        "The make_moons dataset is, by construction, not linearly separable: the two classes occupy "
        "crescent-shaped regions that interleave in 2-dimensional space. Because the Linear kernel "
        "K(x,z) = x·z induces a hyperplane in the original feature space, the resulting boundary "
        "is a straight line that inevitably misclassifies points in the overlapping region near the "
        "tips of the crescents. Increasing the regularisation parameter C makes the margin harder "
        "but does not change the fundamental shape of the boundary. The only path to correct "
        "classification of such data with SVM is to employ a kernel that can map the data into a "
        "higher-dimensional space where the classes become linearly separable — precisely what the "
        "Polynomial and RBF kernels achieve."
    )

    add_heading(doc, "7.2 Performance Convergence on the Breast Cancer Dataset", 2)
    add_paragraph(doc,
        "In contrast to the synthetic dataset, all three kernels achieve high accuracy (0.965–0.983) "
        "on the Breast Cancer Wisconsin dataset. This convergence is explained by the near-linear "
        "separability of the data in its full 30-dimensional feature space. When data is "
        "approximately linearly separable in the original feature space, the Linear kernel already "
        "finds a near-optimal boundary, and the additional flexibility of the Polynomial and RBF "
        "kernels provides only marginal improvement. The Polynomial kernel (d=3) performs "
        "marginally below the Linear kernel on this dataset (0.965 vs 0.974), which suggests that "
        "the higher-order feature interactions introduced by the degree-3 polynomial are not only "
        "unnecessary but slightly counterproductive — possibly introducing mild overfitting or "
        "a less optimal margin orientation. The RBF kernel nonetheless achieves the best "
        "performance (0.983), demonstrating its ability to find the globally optimal boundary "
        "regardless of the data's intrinsic dimensionality."
    )

    add_heading(doc, "7.3 RBF as the Most Robust General-Purpose Kernel", 2)
    add_paragraph(doc,
        "Across both datasets, the RBF kernel achieves the highest or equal-highest F1-score, "
        "validating its reputation as the most robust general-purpose SVM kernel. The fundamental "
        "reason is that the Gaussian kernel K(x,z) = exp(−γ||x−z||²) corresponds to a mapping into "
        "an infinite-dimensional Hilbert space, which provides sufficient representational capacity "
        "to model any continuous decision boundary given appropriate hyperparameters. By tuning γ, "
        "one can smoothly interpolate between highly global (small γ) and highly local (large γ) "
        "decision functions, and by tuning C, one can control the trade-off between margin width "
        "and misclassification tolerance. GridSearchCV identified C = 10, γ = 0.1 as the optimal "
        "combination for the make_moons dataset, yielding a well-calibrated boundary that "
        "generalises effectively. For practitioners, the RBF kernel with cross-validated "
        "hyperparameter tuning is the recommended default choice for SVM-based binary "
        "classification unless domain knowledge or computational constraints strongly favour "
        "an alternative."
    )

    add_page_break(doc)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 8 — CONCLUSION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "8. Conclusion", 1)
    add_paragraph(doc,
        "This project has conducted a systematic empirical comparison of three SVM kernel functions "
        "— Linear, Polynomial (degree=3), and RBF — across a synthetic non-linearly separable "
        "dataset (make_moons) and a real-world biomedical classification dataset (Breast Cancer "
        "Wisconsin). The principal findings are as follows."
    )
    conclusions = [
        "The RBF kernel consistently achieves the best classification performance on both datasets, "
        "attaining an F1-score of 0.950 on the synthetic data and 0.983 on the real-world data. "
        "Its ability to model arbitrarily complex decision boundaries through the Gaussian kernel "
        "function makes it the most versatile and reliable choice for binary SVM classification.",
        "The Linear kernel, whilst computationally the most efficient, is fundamentally limited to "
        "linearly separable problems. On the make_moons dataset, it achieves only 0.830 F1-score — "
        "a gap of 0.12 compared to RBF — because no straight hyperplane can correctly partition "
        "the two interleaving crescent-shaped classes.",
        "The Polynomial kernel (degree=3) offers a meaningful improvement over the Linear kernel "
        "on non-linear data, achieving approximately 0.880 F1-score on make_moons, but does not "
        "fully capture the complexity of the crescent boundary. It is best suited for data with "
        "moderate, polynomial-structured non-linearity.",
        "The gamma parameter of the RBF kernel exerts a strong influence on boundary complexity. "
        "GridSearchCV identified C = 10, γ = 0.1 as the optimal configuration, yielding a "
        "boundary that generalises effectively without overfitting. The gamma effect analysis "
        "confirms that values that are too small (γ = 0.01) lead to underfitting, while values "
        "that are too large (γ = 100) cause severe overfitting with 361 support vectors.",
    ]
    for item in conclusions:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(item).font.size = Pt(11)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.space_after = Pt(5)
    doc.add_paragraph()

    add_paragraph(doc,
        "In conclusion, when the structure of the data is unknown a priori, the RBF kernel "
        "with cross-validated hyperparameter tuning is the recommended default kernel for "
        "SVM-based classification. Future work could extend this comparison to multi-class "
        "classification problems, larger datasets, and a broader set of kernel functions "
        "including the sigmoid and custom user-defined kernels."
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 9 — REFERENCES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    add_heading(doc, "9. References", 1)
    references = [
        "[1] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine "
        "Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. "
        "Available at: https://scikit-learn.org/stable/",
        "[2] Scikit-learn Documentation. Support Vector Machines. "
        "Available at: https://scikit-learn.org/stable/modules/svm.html (Accessed: September 2026).",
        "[3] Scikit-learn Documentation. sklearn.svm.SVC — C-Support Vector Classification. "
        "Available at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
        "[4] Scikit-learn Documentation. sklearn.model_selection.GridSearchCV. "
        "Available at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html",
        "[5] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.",
        "[6] Schölkopf, B., & Smola, A. J. (2002). Learning with Kernels: Support Vector Machines, "
        "Regularization, Optimization, and Beyond. MIT Press.",
        "[7] Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin "
        "(Diagnostic) Data Set. UCI Machine Learning Repository. "
        "Available at: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
        "[8] Course Unit Material: Kernel Functions and Support Vector Machines. "
        "Machine Learning, Department of Computer Science & Engineering, [Your University], 2025–2026.",
    ]
    for ref in references:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(ref).font.size = Pt(10)
        p.paragraph_format.left_indent = Cm(0.0)
        p.paragraph_format.space_after = Pt(5)

    # ── Save ─────────────────────────────────────────────────────────────────
    doc.save(OUTPUT_DOCX)
    print(f"\n✅ Report saved: {OUTPUT_DOCX}")
    size_kb = os.path.getsize(OUTPUT_DOCX) // 1024
    print(f"   File size: {size_kb} KB")
    return OUTPUT_DOCX


if __name__ == "__main__":
    build_report()
