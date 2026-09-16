# AI-Based Network Intrusion Detection System
# 基於機器學習的網路入侵偵測系統

本專案使用 **UNSW-NB15** 網路入侵資料集，建立基於機器學習（Machine Learning）的網路入侵偵測系統（Intrusion Detection System, IDS）原型。

目前系統以 **Binary Classification（二元分類）** 為主要任務，將 Network Flow 分為：

- `0`：Normal（正常流量）
- `1`：Attack（攻擊流量）

目前已完成 Logistic Regression、Random Forest 與 XGBoost 三種模型的訓練與比較，並以 XGBoost 建立可保存及重新載入的 IDS 模型，可對符合 UNSW-NB15 Feature Schema 的外部 CSV Network Flow 資料進行推論。

---

## 專案目標

本專案希望探討機器學習應用於 Network Intrusion Detection 時的以下問題：

1. 不同機器學習模型在入侵偵測上的表現差異。
2. False Positive（誤報）與 False Negative（漏報）對 IDS 的影響。
3. Classification Threshold 對攻擊偵測率與誤報率的影響。
4. 如何將模型訓練流程與實際推論流程分離。
5. 如何建立可接受新 Network Flow 資料的 IDS Prototype。

---

## Dataset

本專案使用 **UNSW-NB15 Dataset**。

目前使用：

| Dataset | Samples | Columns |
|---|---:|---:|
| Training Set | 175,341 | 45 |
| Testing Set | 82,332 | 45 |

原始資料包含：

- Network Flow Features
- `attack_cat`
- `label`
- `id`

模型訓練時移除：

```text
id
attack_cat
label
```

因此實際輸入模型的原始 Features 為：

```text
42 Features
```

其中：

```text
3 Categorical Features
39 Numerical Features
```

Categorical Features：

```text
proto
service
state
```

經過 One-Hot Encoding 後：

```text
42 Original Features
        ↓
One-Hot Encoding
        ↓
194 Processed Features
```

---

## 避免 Data Leakage

`attack_cat` 表示攻擊類型，與最終 Binary Classification Label 有直接關聯。

如果將 `attack_cat` 作為模型輸入，可能讓模型在訓練過程中取得不應存在於實際推論階段的答案資訊，因此本專案將其從 Model Features 中移除。

模型只使用 42 個 Network Flow Features 進行 Normal / Attack 判斷。

另外，資料前處理流程遵循：

```text
Training Set
    ↓
fit_transform()

Testing / New Data
    ↓
transform()
```

避免使用 Test Set 資訊 fitting Preprocessor。

---

# Models

目前比較三種 Machine Learning Models：

## 1. Logistic Regression

作為本專案的 Linear Baseline Model。

Preprocessing：

```text
Categorical Features
→ One-Hot Encoding

Numerical Features
→ StandardScaler
```

測試結果：

```text
Accuracy: 80.97%

False Positive: 14,419
False Negative: 1,245
```

---

## 2. Random Forest

使用 Tree-based Ensemble Learning。

Preprocessing：

```text
Categorical Features
→ One-Hot Encoding

Numerical Features
→ Passthrough
```

Tree-based Models 不依賴特徵尺度，因此未使用 StandardScaler。

測試結果：

```text
Accuracy: 87.10%

False Positive: 9,963
False Negative: 659
```

---

## 3. XGBoost

使用 Gradient Boosting Tree 建立第三個 IDS Model。

主要設定：

```text
n_estimators = 100
max_depth = 6
learning_rate = 0.1
objective = binary:logistic
```

測試結果：

```text
Accuracy: 87.26%

False Positive: 9,896
False Negative: 592
```

---

# Model Comparison

| Model | Accuracy | False Positive | False Negative |
|---|---:|---:|---:|
| Logistic Regression | 80.97% | 14,419 | 1,245 |
| Random Forest | 87.10% | 9,963 | 659 |
| XGBoost | **87.26%** | **9,896** | **592** |

在目前的初始模型設定下，XGBoost 的測試結果略高於 Random Forest。

但 Random Forest 與 XGBoost 的差距較小，因此目前結果不代表 XGBoost 在所有設定或資料情境下一定優於 Random Forest。

---

# XGBoost Evaluation

XGBoost 在 Test Set 上：

```text
Accuracy : 0.8726
ROC-AUC  : 0.9845
PR-AUC   : 0.9886

FPR      : 0.2675
FNR      : 0.0131
```

Confusion Matrix：

```text
                Predicted
               Normal   Attack

Actual Normal   27104     9896
Actual Attack     592    44740
```

因此：

```text
TN = 27,104
FP =  9,896
FN =    592
TP = 44,740
```

模型具有較高的 Attack Recall，但同時存在較高的 False Positive Rate。

這表示目前 IDS 傾向積極偵測攻擊，能降低漏掉攻擊的比例，但可能造成較多 False Alarm。

---

# Classification Threshold Experiment

XGBoost 可以輸出每筆 Network Flow 屬於 Attack 的 Probability。

目前 Binary Classification Baseline 使用：

```text
Attack Probability >= 0.5
→ ATTACK

Attack Probability < 0.5
→ NORMAL
```

為研究 Threshold 對 IDS 的影響，本專案測試：

| Threshold | Accuracy | FPR | FNR | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 81.61% | 40.71% | 0.18% | 15,061 | 81 |
| 0.40 | 83.86% | 35.25% | 0.54% | 13,043 | 244 |
| 0.50 | 87.26% | 26.75% | 1.31% | 9,896 | 592 |
| 0.60 | 90.19% | 18.74% | 2.52% | 6,934 | 1,143 |
| 0.70 | 91.58% | 13.46% | 4.30% | 4,981 | 1,948 |
| 0.80 | 92.66% | 7.80% | 6.97% | 2,885 | 3,158 |
| 0.90 | 92.89% | 2.35% | 10.99% | 871 | 4,980 |

實驗顯示：

```text
Threshold ↑
     ↓
False Positive ↓
False Negative ↑
```

因此 Classification Threshold 並不存在單純「越高越好」或「越低越好」的答案。

實際 IDS 需要根據 False Alarm 與 Missed Attack 的成本進行取捨。

目前 Threshold Experiment 是使用 Test Set 進行分析，尚未將其作為正式 Threshold Selection。

後續將考慮使用獨立 Validation Set 進行 Threshold Selection，避免利用 Test Set 進行模型決策。

---

# Feature Importance

Random Forest Feature Importance 中排名較高的 Features 包括：

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | ct_state_ttl | 0.108090 |
| 2 | sttl | 0.095746 |
| 3 | ackdat | 0.052744 |
| 4 | sload | 0.049815 |
| 5 | dload | 0.048782 |
| 6 | dttl | 0.044859 |
| 7 | rate | 0.043624 |
| 8 | dinpkt | 0.041091 |
| 9 | dmean | 0.034943 |
| 10 | sinpkt | 0.034467 |

Feature Importance 表示 Feature 對目前 Random Forest 分裂決策的相對貢獻程度，不代表因果關係。

---

# Model Persistence

完成 XGBoost Training 後，系統會將：

```text
Preprocessor
XGBoost Model
Feature Column Names
```

一起保存至：

```text
models/xgboost_ids.joblib
```

因此進行新資料預測時，不需要重新訓練模型。

整體流程：

```text
Training
────────────────────────

UNSW-NB15 Training Set
        ↓
Preprocessing
        ↓
XGBoost Training
        ↓
xgboost_ids.joblib


Inference
────────────────────────

New Network Flow
        ↓
Load xgboost_ids.joblib
        ↓
Preprocessing
        ↓
XGBoost
        ↓
Attack Probability
        ↓
NORMAL / ATTACK
```

---

# CSV Inference

目前 IDS 可以讀取符合 UNSW-NB15 Feature Schema 的外部 CSV。

CSV 必須包含模型需要的：

```text
42 Network Flow Features
```

不需要：

```text
label
attack_cat
```

執行：

```bash
python src/predict_csv.py data/sample_input.csv
```

輸出範例：

```text
===== AI-IDS Prediction Results =====

Flow #1
Attack Probability : 0.8830
Prediction         : ATTACK

Flow #2
Attack Probability : 0.0007
Prediction         : NORMAL

===== Summary =====

Total Flows : 4
Normal      : 2
Attack      : 2
```

---

# Project Structure

```text
AI-IDS/
│
├── data/
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   └── sample_input.csv
│
├── docs/
│   ├── day01.md
│   └── day02.md
│
├── models/
│   └── xgboost_ids.joblib
│
├── results/
│
├── src/
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── train_logistic.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── predict.py
│   ├── predict_csv.py
│   └── create_sample_input.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> Dataset CSV 與已訓練 Model 不包含於 Git Repository。

---

# Installation

建立 Python Virtual Environment 後：

```bash
pip install -r requirements.txt
```

目前主要 Dependencies：

```text
pandas
numpy
scikit-learn
matplotlib
xgboost
```

---

# 執行方式

## Dataset Exploration

```bash
python src/explore_data.py
```

## Logistic Regression

```bash
python src/train_logistic.py
```

## Random Forest

```bash
python src/train_random_forest.py
```

## XGBoost

```bash
python src/train_xgboost.py
```

## 單筆推論分析

```bash
python src/predict.py
```

## 外部 CSV 推論

```bash
python src/predict_csv.py data/sample_input.csv
```

---

# 目前限制

目前專案屬於 **Machine Learning-based Network Intrusion Detection Prototype**。

目前輸入資料必須已經符合 UNSW-NB15 的 42-Feature Schema。

尚未實作：

```text
Real-time Packet Capture
Flow Reconstruction
Real-time Feature Extraction
Real-time Network Monitoring
```

因此目前系統不能直接從 Network Interface 擷取封包並即時產生 UNSW-NB15 Features。

---

# Future Work

後續預計進行：

- Validation Set 與 Classification Threshold Selection
- ROC Curve / Precision-Recall Curve 視覺化
- XGBoost Feature Explainability
- False Positive / False Negative Analysis
- 儲存 Prediction Results
- IDS Dashboard / CLI 改善
- Network Flow Feature Extraction
- Real-time Network Traffic Detection
- Multi-class Attack Classification

---

# Development Status

目前已完成：

- [x] Dataset Exploration
- [x] Data Preprocessing
- [x] Logistic Regression Baseline
- [x] Random Forest
- [x] XGBoost
- [x] Model Comparison
- [x] Feature Importance
- [x] ROC-AUC / PR-AUC
- [x] FPR / FNR Analysis
- [x] Classification Threshold Experiment
- [x] Model Persistence
- [x] External CSV Inference
- [ ] Validation-based Threshold Selection
- [ ] Visualization
- [ ] Explainability
- [ ] IDS Dashboard
- [ ] Real-time Feature Extraction