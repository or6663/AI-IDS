import sys
import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# 1. Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_ids.joblib"


# ==========================================
# 2. Check Command Line Argument
# ==========================================

if len(sys.argv) != 2:
    print("使用方式：")
    print("python src/predict_csv.py <csv_file>")
    sys.exit(1)

input_path = Path(sys.argv[1])

if not input_path.exists():
    print(f"錯誤：找不到檔案 {input_path}")
    sys.exit(1)


# ==========================================
# 3. Load Trained IDS
# ==========================================

print("正在載入 IDS Model...")

ids_pipeline = joblib.load(MODEL_PATH)

preprocessor = ids_pipeline["preprocessor"]
model = ids_pipeline["model"]
feature_columns = ids_pipeline["feature_columns"]

print("IDS Model 載入完成！")


# ==========================================
# 4. Load New Network Flow Data
# ==========================================

print(f"\n正在讀取：{input_path}")

input_df = pd.read_csv(input_path)

print(f"讀取完成，共 {len(input_df)} 筆 Network Flow")


# ==========================================
# 5. Validate Input Features
# ==========================================

missing_columns = [
    column
    for column in feature_columns
    if column not in input_df.columns
]

if missing_columns:
    print("\n錯誤：輸入資料缺少以下 Features：")

    for column in missing_columns:
        print(f"- {column}")

    sys.exit(1)


# 只取模型需要的 42 個 Features
X_input = input_df[feature_columns]


# ==========================================
# 6. Preprocessing
# ==========================================

X_processed = preprocessor.transform(X_input)


# ==========================================
# 7. Prediction
# ==========================================

attack_probabilities = model.predict_proba(
    X_processed
)[:, 1]

# 目前維持 baseline threshold
threshold = 0.5

predictions = (
    attack_probabilities >= threshold
).astype(int)


# ==========================================
# 8. Build Results
# ==========================================

results = pd.DataFrame({
    "attack_probability": attack_probabilities,
    "prediction": [
        "ATTACK" if prediction == 1 else "NORMAL"
        for prediction in predictions
    ]
})


# ==========================================
# 9. Display Results
# ==========================================

print("\n===== AI-IDS Prediction Results =====")

for index, row in results.iterrows():

    print(f"\nFlow #{index + 1}")
    print(
        f"Attack Probability : "
        f"{row['attack_probability']:.4f}"
    )
    print(
        f"Prediction         : "
        f"{row['prediction']}"
    )

print("\n===== Summary =====")

normal_count = (results["prediction"] == "NORMAL").sum()
attack_count = (results["prediction"] == "ATTACK").sum()

print(f"Total Flows : {len(results)}")
print(f"Normal      : {normal_count}")
print(f"Attack      : {attack_count}")