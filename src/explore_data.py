import pandas as pd
from pathlib import Path


# ==========================================
# 1. 設定資料集路徑
# ==========================================

# __file__ 是目前這支 explore_data.py 的位置
# parent.parent 會從 src/ 回到 AI-IDS/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"


# ==========================================
# 2. 讀取 CSV
# ==========================================

print("正在讀取 UNSW-NB15 Dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("資料讀取完成！")


# ==========================================
# 3. 查看資料大小
# ==========================================

print("\n===== Dataset Shape =====")

print("Training set:", train_df.shape)
print("Testing set :", test_df.shape)


# ==========================================
# 4. 查看前 5 筆資料
# ==========================================

print("\n===== First 5 Rows =====")

print(train_df.head())


# ==========================================
# 5. 查看所有欄位
# ==========================================

print("\n===== Columns =====")

for i, column in enumerate(train_df.columns):
    print(f"{i}: {column}")


# ==========================================
# 6. 查看 Normal / Attack 數量
# ==========================================

print("\n===== Binary Label Distribution =====")

print(train_df["label"].value_counts())


# ==========================================
# 7. 查看各種攻擊類型
# ==========================================

print("\n===== Attack Category Distribution =====")

print(train_df["attack_cat"].value_counts())


# ==========================================
# 8. 查看資料型態
# ==========================================

print("\n===== Data Types =====")

print(train_df.dtypes)


# ==========================================
# 9. 查看缺失值
# ==========================================

print("\n===== Missing Values =====")

missing_values = train_df.isnull().sum()

# 只顯示真的存在缺失值的欄位
print(missing_values[missing_values > 0])