import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "UNSW_NB15_testing-set.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_input.csv"
)


# 讀取原始 Test Dataset
test_df = pd.read_csv(TEST_PATH)


# 選擇幾筆資料作為 Demo
sample_indices = [0, 8, 243, 254]

sample_df = test_df.iloc[sample_indices].copy()


# 移除模型不應該知道的資訊
sample_df = sample_df.drop(
    columns=[
        "id",
        "attack_cat",
        "label",
    ]
)


# 輸出成新的 CSV
sample_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("Sample input 建立完成！")
print("Output:", OUTPUT_PATH)

print("\nShape:", sample_df.shape)

print("\nColumns:")
print(sample_df.columns.tolist())