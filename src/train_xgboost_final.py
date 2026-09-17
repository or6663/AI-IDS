import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from pathlib import Path
from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ==========================================
# 1. Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "UNSW_NB15_training-set.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "UNSW_NB15_testing-set.csv"
)


# ==========================================
# 2. Load Dataset
# ==========================================

print("正在讀取 UNSW-NB15 Dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("資料讀取完成！")

print("\n===== Original Dataset =====")
print(f"Original Training Set : {train_df.shape}")
print(f"Testing Set           : {test_df.shape}")


# ==========================================
# 3. Separate Features and Label
# ==========================================

DROP_COLUMNS = [
    "id",
    "attack_cat",
    "label",
]

X = train_df.drop(
    columns=DROP_COLUMNS
)

y = train_df["label"]


# ==========================================
# 4. Train / Validation Split
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ==========================================
# 5. Prepare Final Test Set
# ==========================================

X_test = test_df.drop(
    columns=DROP_COLUMNS
)

y_test = test_df["label"]


# ==========================================
# 6. Display Split Information
# ==========================================

print("\n===== Train / Validation / Test =====")

print(
    f"Training Set   : "
    f"{X_train.shape}"
)

print(
    f"Validation Set : "
    f"{X_val.shape}"
)

print(
    f"Testing Set    : "
    f"{X_test.shape}"
)


print("\n===== Training Label Distribution =====")
print(y_train.value_counts())

print("\n===== Validation Label Distribution =====")
print(y_val.value_counts())

print("\n===== Testing Label Distribution =====")
print(y_test.value_counts())


# ==========================================
# 7. Identify Feature Types
# ==========================================

categorical_columns = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\n===== Feature Types =====")
print(
    f"Categorical Features : "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical Features   : "
    f"{len(numerical_columns)}"
)


# ==========================================
# 8. Build Preprocessor
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns,
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns,
        ),
    ]
)


# ==========================================
# 9. Fit Preprocessor ONLY on Training Data
# ==========================================

print("\n正在進行 Preprocessing...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)

print("Preprocessing 完成！")

print("\n===== Processed Shapes =====")
print(
    f"Training   : "
    f"{X_train_processed.shape}"
)

print(
    f"Validation : "
    f"{X_val_processed.shape}"
)


# ==========================================
# 10. Train XGBoost
# ==========================================

print("\n正在訓練 XGBoost...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train_processed,
    y_train,
)

print("XGBoost 訓練完成！")


# ==========================================
# 11. Validation Prediction
# ==========================================

val_probabilities = model.predict_proba(
    X_val_processed
)[:, 1]


print("\n===== Validation Probability =====")
print(
    f"Minimum : "
    f"{val_probabilities.min():.4f}"
)

print(
    f"Maximum : "
    f"{val_probabilities.max():.4f}"
)

print(
    f"Mean    : "
    f"{val_probabilities.mean():.4f}"
)


# ==========================================
# 12. Validation Threshold Analysis
# ==========================================




print("\n===== Validation Threshold Analysis =====")

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Recall':<12}"
    f"{'FPR':<12}"
    f"{'FP':<10}"
    f"{'FN':<10}"
)


validation_results = []


for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions
    ).ravel()

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn)
    )

    attack_recall = (
        tp
        / (tp + fn)
    )

    fpr = (
        fp
        / (fp + tn)
    )

    validation_results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "recall": attack_recall,
        "fpr": fpr,
        "fp": fp,
        "fn": fn,
    })

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{attack_recall:<12.4f}"
        f"{fpr:<12.4f}"
        f"{fp:<10}"
        f"{fn:<10}"
    )


# ==========================================
# 13. Lock Final Threshold
# ==========================================

CHOSEN_THRESHOLD = 0.45

print("\n===== Selected Threshold =====")
print(f"Final Threshold : {CHOSEN_THRESHOLD:.2f}")

print(
    "Selection Rule : "
    "Attack Recall >= 98%, "
    "then minimize FPR"
)


# ==========================================
# 14. Transform Final Test Set
# ==========================================

print("\n正在處理 Final Test Set...")

X_test_processed = preprocessor.transform(
    X_test
)

print(
    f"Testing : "
    f"{X_test_processed.shape}"
)


# ==========================================
# 15. Final Test Prediction
# ==========================================

test_probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

test_predictions = (
    test_probabilities >= CHOSEN_THRESHOLD
).astype(int)


# ==========================================
# 16. Final Test Evaluation
# ==========================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions
).ravel()

accuracy = accuracy_score(
    y_test,
    test_predictions
)

attack_recall = (
    tp / (tp + fn)
)

fpr = (
    fp / (fp + tn)
)

fnr = (
    fn / (fn + tp)
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


print("\n===== FINAL TEST RESULTS =====")

print(
    f"Threshold     : "
    f"{CHOSEN_THRESHOLD:.2f}"
)

print(
    f"Accuracy      : "
    f"{accuracy:.4f}"
)

print(
    f"Attack Recall : "
    f"{attack_recall:.4f}"
)

print(
    f"FPR           : "
    f"{fpr:.4f}"
)

print(
    f"FNR           : "
    f"{fnr:.4f}"
)

print(
    f"ROC-AUC       : "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC        : "
    f"{pr_auc:.4f}"
)


print("\n===== Confusion Matrix =====")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

print(f"\nTN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")
print(f"TP : {tp}")


print("\n===== Classification Report =====")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Normal",
            "Attack",
        ],
        digits=4,
    )
)


# ==========================================
# 17. Create Results Directory
# ==========================================



RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    f"\nResults Directory : "
    f"{RESULTS_DIR}"
)


# ==========================================
# 18. Plot Confusion Matrix
# ==========================================

print("\n正在產生 Confusion Matrix...")

cm = confusion_matrix(
    y_test,
    test_predictions,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Attack",
    ],
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

display.plot(
    ax=ax,
    values_format="d",
)

ax.set_title(
    "XGBoost IDS - Confusion Matrix"
)

plt.tight_layout()


CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "confusion_matrix.png"
)

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Confusion Matrix saved to:"
)

print(
    CONFUSION_MATRIX_PATH
)

# ==========================================
# 19. Plot ROC Curve
# ==========================================

print("\n正在產生 ROC Curve...")

fpr_values, tpr_values, _ = roc_curve(
    y_test,
    test_probabilities,
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.plot(
    fpr_values,
    tpr_values,
    label=f"XGBoost (AUC = {roc_auc:.4f})",
)

# Random classifier baseline
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "XGBoost IDS - ROC Curve"
)

ax.legend(
    loc="lower right"
)

ax.grid(
    alpha=0.3
)

plt.tight_layout()


ROC_CURVE_PATH = (
    RESULTS_DIR
    / "roc_curve.png"
)

plt.savefig(
    ROC_CURVE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "ROC Curve saved to:"
)

print(
    ROC_CURVE_PATH
)


# ==========================================
# 20. Plot Precision-Recall Curve
# ==========================================

print("\n正在產生 Precision-Recall Curve...")

precision_values, recall_values, _ = (
    precision_recall_curve(
        y_test,
        test_probabilities,
    )
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.plot(
    recall_values,
    precision_values,
    label=f"XGBoost (AP = {pr_auc:.4f})",
)

ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.set_title(
    "XGBoost IDS - Precision-Recall Curve"
)

ax.legend(
    loc="lower left"
)

ax.grid(
    alpha=0.3
)

plt.tight_layout()


PR_CURVE_PATH = (
    RESULTS_DIR
    / "pr_curve.png"
)

plt.savefig(
    PR_CURVE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Precision-Recall Curve saved to:"
)

print(
    PR_CURVE_PATH
)


# ==========================================
# 21. Plot Validation Threshold Trade-off
# ==========================================

print("\n正在產生 Threshold Trade-off 圖...")

threshold_values = [
    result["threshold"]
    for result in validation_results
]

recall_values_threshold = [
    result["recall"]
    for result in validation_results
]

fpr_values_threshold = [
    result["fpr"]
    for result in validation_results
]


fig, ax = plt.subplots(
    figsize=(8, 6)
)

# Attack Recall
ax.plot(
    threshold_values,
    recall_values_threshold,
    marker="o",
    label="Attack Recall",
)

# False Positive Rate
ax.plot(
    threshold_values,
    fpr_values_threshold,
    marker="o",
    label="False Positive Rate",
)

# 98% Recall requirement
ax.axhline(
    y=0.98,
    linestyle="--",
    label="Minimum Recall = 98%",
)

# Selected threshold
ax.axvline(
    x=CHOSEN_THRESHOLD,
    linestyle="--",
    label=f"Selected Threshold = {CHOSEN_THRESHOLD:.2f}",
)

ax.set_xlabel(
    "Classification Threshold"
)

ax.set_ylabel(
    "Rate"
)

ax.set_title(
    "Validation Threshold Trade-off"
)

ax.set_ylim(
    0,
    1.05,
)

ax.legend()

ax.grid(
    alpha=0.3
)

plt.tight_layout()


THRESHOLD_PLOT_PATH = (
    RESULTS_DIR
    / "threshold_tradeoff.png"
)

plt.savefig(
    THRESHOLD_PLOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Threshold Trade-off saved to:"
)

print(
    THRESHOLD_PLOT_PATH
)


# ==========================================
# 22. Error Analysis
# ==========================================

print("\n===== Error Analysis =====")

error_df = test_df.copy()

error_df["attack_probability"] = (
    test_probabilities
)

error_df["prediction"] = (
    test_predictions
)


# False Positive
false_positives = error_df[
    (error_df["label"] == 0)
    & (error_df["prediction"] == 1)
]


# False Negative
false_negatives = error_df[
    (error_df["label"] == 1)
    & (error_df["prediction"] == 0)
]


print(
    f"False Positives : "
    f"{len(false_positives)}"
)

print(
    f"False Negatives : "
    f"{len(false_negatives)}"
)


# ==========================================
# 23. False Negative Attack Categories
# ==========================================

print(
    "\n===== False Negative Attack Categories ====="
)

fn_attack_categories = (
    false_negatives["attack_cat"]
    .value_counts()
)

print(
    fn_attack_categories
)


# ==========================================
# 24. Error Probability Statistics
# ==========================================

print(
    "\n===== False Positive Probability ====="
)

print(
    false_positives[
        "attack_probability"
    ].describe()
)


print(
    "\n===== False Negative Probability ====="
)

print(
    false_negatives[
        "attack_probability"
    ].describe()
)

# ==========================================
# 25. False Negative Rate by Attack Category
# ==========================================

print(
    "\n===== FN Rate by Attack Category ====="
)

attack_test_data = error_df[
    error_df["label"] == 1
]

attack_category_total = (
    attack_test_data["attack_cat"]
    .value_counts()
)

attack_category_fn = (
    false_negatives["attack_cat"]
    .value_counts()
)

fn_rate_by_category = pd.DataFrame({
    "Total Attacks": attack_category_total,
    "False Negatives": attack_category_fn,
})

fn_rate_by_category[
    "False Negatives"
] = (
    fn_rate_by_category[
        "False Negatives"
    ]
    .fillna(0)
    .astype(int)
)

fn_rate_by_category[
    "FN Rate"
] = (
    fn_rate_by_category[
        "False Negatives"
    ]
    /
    fn_rate_by_category[
        "Total Attacks"
    ]
)

fn_rate_by_category = (
    fn_rate_by_category
    .sort_values(
        by="FN Rate",
        ascending=False,
    )
)

print(
    fn_rate_by_category
)


# ==========================================
# 26. Save Final IDS Model
# ==========================================

print("\n正在儲存 Final IDS Model...")

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "xgboost_ids_final.joblib"
)


final_ids_pipeline = {
    "preprocessor": preprocessor,
    "model": model,
    "feature_columns": X_train.columns.tolist(),
    "threshold": CHOSEN_THRESHOLD,
}


joblib.dump(
    final_ids_pipeline,
    FINAL_MODEL_PATH,
)


print(
    "Final IDS Model saved to:"
)

print(
    FINAL_MODEL_PATH
)