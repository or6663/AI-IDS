import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. 設定 Dataset 路徑
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"


# ==========================================
# 2. 讀取 Dataset
# ==========================================

print("正在讀取 Dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Dataset 讀取完成！")


# ==========================================
# 3. 分離 Features 與 Label
# ==========================================

DROP_COLUMNS = ["id", "attack_cat", "label"]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["label"]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"]


# ==========================================
# 4. 找出類別型與數值型 Features
# ==========================================

categorical_columns = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical features:", len(categorical_columns))
print("Numerical features  :", len(numerical_columns))


# ==========================================
# 5. 資料前處理
# ==========================================

# XGBoost 是 Tree-based Model
# Numerical Features 不需要 StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns,
        ),
    ]
)

print("\n正在進行資料前處理...")

X_train_processed = preprocessor.fit_transform(X_train)

# Test Set 只能 transform，避免 Data Leakage
X_test_processed = preprocessor.transform(X_test)

print("資料前處理完成！")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape :", X_test_processed.shape)


# ==========================================
# 6. 建立 XGBoost IDS
# ==========================================

print("\n正在訓練 XGBoost IDS...")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

print("模型訓練完成！")


# ==========================================
# 7. Test Set 預測
# ==========================================

print("\n正在測試 IDS...")

y_pred = model.predict(X_test_processed)


# ==========================================
# 8. 評估模型
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
# 取得每筆資料被判定為 Attack (class 1) 的機率
y_prob = model.predict_proba(X_test_processed)[:, 1]

print("\n===== XGBoost IDS Result =====")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"]
    )
)


# ==========================================
# 9. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n===== Confusion Matrix =====")

print(cm)

print("\nDetailed Results:")
print("True Negative  (Normal -> Normal):", tn)
print("False Positive (Normal -> Attack):", fp)
print("False Negative (Attack -> Normal):", fn)
print("True Positive  (Attack -> Attack):", tp)

# ==========================================
# 10. Additional IDS Evaluation Metrics
# ==========================================

# False Positive Rate
fpr = fp / (fp + tn)

# False Negative Rate
fnr = fn / (fn + tp)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

# PR-AUC
pr_auc = average_precision_score(y_test, y_prob)


print("\n===== Additional IDS Metrics =====")

print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"False Negative Rate (FNR): {fnr:.4f}")
print(f"ROC-AUC                  : {roc_auc:.4f}")
print(f"PR-AUC                   : {pr_auc:.4f}")

# ==========================================
# 11. Classification Threshold Experiment
# ==========================================

print("\n===== Threshold Experiment =====")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'FPR':<12}"
    f"{'FNR':<12}"
    f"{'FP':<10}"
    f"{'FN':<10}"
)

for threshold in thresholds:

    # 根據指定 threshold 自己產生分類結果
    y_pred_threshold = (y_prob >= threshold).astype(int)

    # Confusion Matrix
    cm_threshold = confusion_matrix(
        y_test,
        y_pred_threshold
    )

    tn_t, fp_t, fn_t, tp_t = cm_threshold.ravel()

    # Accuracy
    accuracy_t = accuracy_score(
        y_test,
        y_pred_threshold
    )

    # False Positive Rate
    fpr_t = fp_t / (fp_t + tn_t)

    # False Negative Rate
    fnr_t = fn_t / (fn_t + tp_t)

    print(
        f"{threshold:<12.2f}"
        f"{accuracy_t:<12.4f}"
        f"{fpr_t:<12.4f}"
        f"{fnr_t:<12.4f}"
        f"{fp_t:<10}"
        f"{fn_t:<10}"
    )

# ==========================================
# 12. Save Trained IDS
# ==========================================

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_ids.joblib"

# Preprocessor 和 Model 必須一起保存
ids_pipeline = {
    "preprocessor": preprocessor,
    "model": model,
    "feature_columns": X_train.columns.tolist(),
}

joblib.dump(ids_pipeline, MODEL_PATH)

print("\n===== Model Saved =====")
print("Model path:", MODEL_PATH)