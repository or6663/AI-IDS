import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


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

# Random Forest 不需要 StandardScaler
# 類別型資料仍需要 One-Hot Encoding
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
X_test_processed = preprocessor.transform(X_test)

print("資料前處理完成！")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape :", X_test_processed.shape)


# ==========================================
# 6. 建立 Random Forest IDS
# ==========================================

print("\n正在訓練 Random Forest IDS...")

model = RandomForestClassifier(
    n_estimators=100,
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

print("\n===== Random Forest IDS Result =====")

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
# 10. Feature Importance
# ==========================================

print("\n===== Top 15 Feature Importance =====")

# 取得 One-Hot Encoding 後的完整 Feature 名稱
feature_names = preprocessor.get_feature_names_out()

# 取得 Random Forest 計算出的 Feature Importance
importances = model.feature_importances_

# 建立 DataFrame，方便排序與顯示
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Importance 由大到小排序
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# 顯示最重要的前 15 個 Features
print(
    feature_importance_df.head(15).to_string(index=False)
)