import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# 1. Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_ids.joblib"
TEST_PATH = PROJECT_ROOT / "data" / "UNSW_NB15_testing-set.csv"


# ==========================================
# 2. Load Trained IDS
# ==========================================

print("正在載入 IDS Model...")

ids_pipeline = joblib.load(MODEL_PATH)

preprocessor = ids_pipeline["preprocessor"]
model = ids_pipeline["model"]
feature_columns = ids_pipeline["feature_columns"]

print("IDS Model 載入完成！")


# ==========================================
# 3. Load Test Dataset
# ==========================================

print("正在讀取測試資料...")

test_df = pd.read_csv(TEST_PATH)

X_test = test_df[feature_columns]

y_true = test_df["label"].to_numpy()


# ==========================================
# 4. Preprocessing
# ==========================================

X_test_processed = preprocessor.transform(X_test)


# ==========================================
# 5. Prediction
# ==========================================

attack_probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

threshold = 0.5

y_pred = (
    attack_probabilities >= threshold
).astype(int)


# ==========================================
# 6. Find TN / FP / FN / TP Examples
# ==========================================

cases = {
    "True Negative (TN)": (0, 0),
    "False Positive (FP)": (0, 1),
    "False Negative (FN)": (1, 0),
    "True Positive (TP)": (1, 1),
}


print("\n===== AI-IDS Example Predictions =====")

for case_name, (true_value, predicted_value) in cases.items():

    # 找出符合這種情況的 index
    matching_indices = (
        (y_true == true_value) &
        (y_pred == predicted_value)
    ).nonzero()[0]

    if len(matching_indices) == 0:
        print(f"\n{case_name}: 找不到案例")
        continue

    # 取第一個符合條件的案例
    index = matching_indices[0]

    probability = attack_probabilities[index]

    attack_category = test_df.iloc[index]["attack_cat"]

    true_text = (
        "ATTACK"
        if true_value == 1
        else "NORMAL"
    )

    prediction_text = (
        "ATTACK"
        if predicted_value == 1
        else "NORMAL"
    )

    print(f"\n----- {case_name} -----")
    print(f"Sample Index       : {index}")
    print(f"Attack Probability : {probability:.4f}")
    print(f"Prediction         : {prediction_text}")
    print(f"True Label         : {true_text}")
    print(f"Attack Category    : {attack_category}")