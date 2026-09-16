import pandas as pd
from pathlib import Path

# ==========================================
# 1. 設定專案與 Dataset 路徑
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"


# ==========================================
# 2. 讀取資料
# ==========================================

print("正在讀取 Dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Dataset 讀取完成！")


# ==========================================
# 3. 分離 Features (X) 與 Label (y)
# ==========================================

# id 只是資料編號，對判斷攻擊沒有意義
# attack_cat 直接包含攻擊類型，
# 如果拿來預測 label 會造成 Data Leakage
DROP_COLUMNS = ["id", "attack_cat", "label"]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["label"]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"]


# ==========================================
# 4. 找出數值與類別欄位
# ==========================================

categorical_columns = X_train.select_dtypes(include=["object", "str"]).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ==========================================
# 5. 顯示結果
# ==========================================

print("\n===== Preprocessing Information =====")

print("Training features shape:", X_train.shape)
print("Testing features shape :", X_test.shape)

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumber of categorical columns:", len(categorical_columns))
print("Number of numerical columns  :", len(numerical_columns))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nPreprocessing preparation completed!")