# Day 4 — Streamlit IDS Dashboard

## 1. Day 4 目標

Day 4 的主要目標是將 Day 3 已完成並凍結（Frozen）的 AI-IDS 模型整合到可互動的 Web Dashboard，讓使用者可以上傳 network flow CSV，完成輸入驗證、模型推論、結果視覺化與結果下載。

本日不重新訓練模型，也不修改 XGBoost 超參數或正式 classification threshold。

```text
Network Flow CSV
        ↓
42-Feature Schema Validation
        ↓
Frozen Preprocessor
        ↓
Frozen XGBoost Model
        ↓
Attack Probability
        ↓
Frozen Threshold = 0.45
        ↓
NORMAL / ATTACK
        ↓
Dashboard / Alert / Visualization / CSV Export
```

---

## 2. Dashboard 技術選擇

Dashboard 使用 Streamlit 建立，並使用 Altair 製作 Attack Probability 視覺化。

Day 4 新增的直接依賴：

```text
streamlit==1.64.0
altair==6.3.0
```

Streamlit 負責 Web UI、CSV upload、metrics、表格、alert 與結果下載；Altair 用於 probability bar chart 與 threshold line。

---

## 3. Frozen IDS Model 載入

Dashboard 使用 Day 3 儲存的正式模型：

```text
models/xgboost_ids_final.joblib
```

Bundle 包含：

```python
{
    "preprocessor": preprocessor,
    "model": model,
    "feature_columns": feature_columns,
    "threshold": threshold,
}
```

Dashboard 直接讀取：

```python
threshold = ids_pipeline["threshold"]
feature_columns = ids_pipeline["feature_columns"]
preprocessor = ids_pipeline["preprocessor"]
model = ids_pipeline["model"]
```

因此 Dashboard 不重新訓練模型，也不另外 hardcode `0.45`。

成功讀取：

```text
Classification Threshold: 0.45
Required Input Features: 42
```

模型載入函式使用 `@st.cache_resource`，避免 Streamlit UI interaction 重新執行 script 時反覆從磁碟載入模型。

---

## 4. Network Flow CSV Upload

Dashboard 使用 `st.file_uploader()` 接收 CSV。

Day 4 使用 `data/sample_input.csv` 作為正常輸入測試：

```text
4 flows
42 required model features
```

---

## 5. 42-Feature Schema Validation

Dashboard 不直接將任意 CSV 送入模型，而是根據 frozen bundle 中的 `feature_columns` 檢查必要欄位。

```python
missing_features = [
    feature
    for feature in feature_columns
    if feature not in input_data.columns
]
```

若所有必要欄位存在：

```python
model_input = input_data[feature_columns].copy()
```

這可確保：

1. 模型輸入 feature 順序與訓練時一致。
2. 僅保留模型實際需要的 42 features。
3. 額外欄位不會意外進入模型。

若缺少必要 feature，Dashboard 會停止後續 inference 並顯示缺少欄位。

---

## 6. Frozen Preprocessor 與 XGBoost Inference

Schema validation 通過後：

```python
processed_data = preprocessor.transform(model_input)
```

再由 frozen XGBoost model 計算：

```python
attack_probabilities = model.predict_proba(
    processed_data
)[:, 1]
```

其中 Class 0 為 Normal，Class 1 為 Attack，因此 `[:, 1]` 取得每筆 flow 的 Attack Probability。

---

## 7. Classification Threshold

Dashboard 使用 bundle 中儲存的正式 threshold：

```text
0.45
```

分類規則：

```python
predictions = (
    attack_probabilities >= threshold
).astype(int)
```

也就是：

```text
Attack Probability >= 0.45 → ATTACK
Attack Probability <  0.45 → NORMAL
```

Threshold 並非 Day 4 重新選擇，而是沿用 Day 3 validation-based threshold selection 的結果。

---

## 8. Sample Input Inference Verification

四筆 `sample_input.csv` 的 Dashboard 結果：

| Flow | Attack Probability | Prediction |
|---:|---:|---|
| 1 | 0.8939 | ATTACK |
| 2 | 0.0010 | NORMAL |
| 3 | 0.9968 | ATTACK |
| 4 | 0.4755 | ATTACK |

Flow 4 因為：

```text
0.4755 >= 0.45
```

所以分類為 ATTACK。

此結果與先前 `src/predict_csv.py` 使用 frozen model 的 inference 結果一致，確認 CLI inference 與 Streamlit Dashboard 使用相同推論邏輯。

---

## 9. Detection Summary 與 Security Alert

Dashboard 計算：

```text
Total Flows  : 4
Normal Flows : 1
Attack Flows : 3
Attack Ratio : 75.0%
```

Attack Ratio：

```python
attack_ratio = (
    attack_flows / total_flows * 100
    if total_flows > 0
    else 0
)
```

若 `attack_flows > 0`，顯示：

```text
Security Alert: 3 suspicious flow(s) detected.
```

Dashboard 使用 `suspicious flow` 描述模型警報，而不是直接宣稱已證實發生真實攻擊。

---

## 10. Detection Distribution

Dashboard 建立 Normal / Attack 數量分布圖。

Sample input：

```text
Normal = 1
Attack = 3
```

並固定類別顯示順序為：

```text
Normal → Attack
```

---

## 11. Attack Probability Overview

Dashboard 顯示每筆 flow 的 Attack Probability：

```text
Flow 1 = 0.8939
Flow 2 = 0.0010
Flow 3 = 0.9968
Flow 4 = 0.4755
```

使用 Altair bar chart，Y-axis 固定在 `0.0 → 1.0`。

圖中另外加入從 frozen bundle 讀取的：

```text
Threshold = 0.45
```

水平虛線與文字標籤。

因此可直接觀察：

```text
Flow 1 > 0.45 → ATTACK
Flow 2 < 0.45 → NORMAL
Flow 3 > 0.45 → ATTACK
Flow 4 > 0.45 → ATTACK
```

---

## 12. Detection Results Table

Dashboard 顯示：

```text
Flow
Attack Probability
Prediction
```

Probability 在 UI 顯示時 round 到小數第 4 位，例如：

```text
0.8939472 → 0.8939
```

四捨五入只作用於顯示；模型 classification 仍使用原始 probability 與 threshold 比較。

UI 使用：

```text
⚠️ ATTACK
✅ NORMAL
```

增加可讀性。

---

## 13. Detection Result CSV Export

Dashboard 提供 `Download Detection Results` 功能。

輸出 CSV 保留模型輸入的 42 features，並新增：

```text
Attack Probability
Prediction
```

因此輸出共：

```text
42 model features
+ Attack Probability
+ Prediction
= 44 columns
```

CSV 中 Prediction 維持 `ATTACK` / `NORMAL`，不加入 Dashboard UI 的 emoji。

---

## 14. Invalid CSV Test

除了正常輸入測試，Day 4 也實際測試錯誤 schema。

從原本 42 features 中移除：

```text
dur
```

輸入因此只剩 41 features。

Dashboard 成功偵測：

```text
Invalid CSV: 1 required feature(s) are missing.

Missing features:
["dur"]
```

此時不繼續執行 preprocessing、XGBoost inference、summary、visualization 或 detection results。

這確認 schema validation 已實際通過錯誤輸入測試。

---

## 15. Day 4 Dashboard 功能總結

截至 Day 4：

```text
Frozen model loading
        ↓
CSV upload
        ↓
42-feature schema validation
        ↓
Frozen preprocessing
        ↓
XGBoost inference
        ↓
Attack probability
        ↓
Threshold-based classification
        ↓
Detection summary
        ↓
Security alert
        ↓
Normal / Attack distribution
        ↓
Per-flow probability visualization
        ↓
Threshold visualization
        ↓
Detection results table
        ↓
Result CSV download
```

---

## 16. 目前技術限制

目前 Dashboard 接收的是已具有 UNSW-NB15 所需 42 個 model features 的 network-flow CSV。

目前尚未完成：

```text
Raw Packet Capture
        ↓
Flow Reconstruction
        ↓
42-Feature Extraction
        ↓
Dashboard
```

因此目前系統仍應描述為：

```text
Machine Learning-Based Network Intrusion Detection Prototype
```

不能僅因為已有 Web Dashboard 就宣稱為完整 real-time packet-based IDS。

---

## 17. Day 4 結論

Day 4 成功將 Day 3 frozen XGBoost IDS model 從 command-line inference pipeline 整合到 Streamlit Web Dashboard。

Dashboard 沒有重新訓練模型，也沒有修改正式 classification threshold。

正常四筆 sample input 的 Dashboard prediction 與既有 CLI inference 結果一致：

```text
Flow 1 → 0.8939 → ATTACK
Flow 2 → 0.0010 → NORMAL
Flow 3 → 0.9968 → ATTACK
Flow 4 → 0.4755 → ATTACK
```

同時完成 42-feature schema validation，並使用缺少 `dur` 的 41-feature CSV 驗證錯誤輸入能被阻擋。

因此 Day 4 完成從 Frozen ML Model 到 Interactive IDS Analysis Dashboard 的整合。

---

## 18. Day 5 預計方向

Day 5 將研究如何讓目前 CSV-based IDS prototype 更接近實際 network traffic detection pipeline：

```text
Network Traffic
      ↓
Packet / Flow Collection
      ↓
Flow Feature Extraction
      ↓
42 Required Features
      ↓
Frozen AI-IDS Model
```

優先確認 UNSW-NB15 的 42 features 中，哪些可以由現有 packet / flow 工具穩定取得，以及完整 live feature reconstruction 的實作成本。

若無法在目前專題時程內可靠完成完整 42-feature real-time extraction，則採用 recorded / prepared network-flow replay 建立 near-real-time demonstration，並明確標示為 flow replay / simulation，而不宣稱為完整 live packet IDS。
