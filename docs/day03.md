# Day 3 — Validation、Threshold Selection 與 Final Model

## 1. 今日目標

Day 3 的主要目標是建立更嚴謹的模型評估流程。Day 2 雖然已完成 XGBoost 與外部 CSV inference，但 threshold experiment 是直接在 Test Set 上進行，因此 Day 3 改為正式區分 Training、Validation 與 Test。

今日完成：Train/Validation/Test 分離、XGBoost 訓練、Validation Threshold Selection、Final Test Evaluation、Confusion Matrix、ROC Curve、Precision-Recall Curve、Threshold Trade-off、FP/FN Error Analysis、Attack Category FN Rate、Final Model Persistence、External CSV Inference。

---

## 2. Dataset Split

使用 UNSW-NB15 Dataset。官方 Training Set 為 175,341 × 45，官方 Testing Set 為 82,332 × 45。從官方 Training Set 中再切出 20% 作為 Validation Set：

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

| Dataset | Samples | Features |
|---|---:|---:|
| Training | 140,272 | 42 |
| Validation | 35,069 | 42 |
| Testing | 82,332 | 42 |

Training：Attack 95,472 / Normal 44,800。Validation：Attack 23,869 / Normal 11,200。Testing：Attack 45,332 / Normal 37,000。

`stratify=y` 用來維持類別比例；`random_state=42` 用來確保資料切分可重現，42 本身沒有特殊 ML 意義。

---

## 3. Feature Preprocessing

模型輸入共有 42 Features，其中 3 個 categorical features：`proto`、`service`、`state`，以及 39 個 numerical features。

Categorical features 使用：

```python
OneHotEncoder(handle_unknown="ignore")
```

Numerical features 使用 `passthrough`。

Day 2 使用完整官方 Training Set fit encoder，因此得到 194 processed features；Day 3 為避免 Validation Data 參與 preprocessing fitting，只使用 140,272 筆 Training subset fit encoder，因此得到 192 processed features。Training、Validation、Testing 都由同一個 training-fitted preprocessor 轉換，所以最後維度一致為 192。

---

## 4. Data Leakage Prevention

Day 3 遵守：

```text
Training   → preprocessor.fit_transform()
Validation → preprocessor.transform()
Testing    → preprocessor.transform()
```

Validation 與 Test 不參與 Preprocessor fitting。

`id`、`attack_cat`、`label` 都不作為模型輸入。`attack_cat` 僅在 prediction 完成後用於 Ground Truth Error Analysis，因此不構成 Training Data Leakage。

---

## 5. XGBoost Model

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```

模型輸出 Attack Probability，再透過 Classification Threshold 轉換成 `NORMAL / ATTACK`。

---

## 6. Validation Threshold Selection

Day 3 在查看 Validation 結果前先設定 operating constraint：

> Attack Recall 必須 ≥ 98%，並在符合條件的 Threshold 中選擇最低 FPR。

98% 並不是 IDS 通用標準，而是本專案的實驗 operating constraint，用來表示優先降低攻擊漏報。實際部署時應依 False Negative / False Positive 成本、SOC 人力與 Security Policy 調整。

| Threshold | Accuracy | Attack Recall | FPR | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 94.10% | 99.78% | 17.99% | 2015 | 53 |
| 0.35 | 94.55% | 99.52% | 16.04% | 1796 | 114 |
| 0.40 | 94.94% | 99.10% | 13.93% | 1560 | 216 |
| 0.45 | 95.35% | 98.47% | 11.29% | 1265 | 366 |
| 0.50 | 95.43% | 97.62% | 9.23% | 1034 | 568 |
| 0.55 | 95.29% | 96.60% | 7.49% | 839 | 812 |
| 0.60 | 95.10% | 95.62% | 6.01% | 673 | 1046 |
| 0.65 | 94.65% | 94.45% | 4.92% | 551 | 1325 |
| 0.70 | 94.17% | 93.38% | 4.15% | 465 | 1581 |
| 0.75 | 93.45% | 91.83% | 3.11% | 348 | 1950 |
| 0.80 | 92.80% | 90.54% | 2.40% | 269 | 2257 |
| 0.85 | 91.83% | 88.79% | 1.69% | 189 | 2675 |
| 0.90 | 90.45% | 86.40% | 0.90% | 101 | 3247 |

符合 Recall ≥ 98% 的 threshold 為 0.30、0.35、0.40、0.45，其中 0.45 的 FPR 最低，因此正式選擇：

```text
Selected Threshold = 0.45
```

Threshold 在 Validation Set 上選定後鎖定，不再根據 Test Set 修改。

---

## 7. Final Test Evaluation

使用 Threshold = 0.45 對官方 UNSW-NB15 Testing Set 進行最終評估。

| Metric | Result |
|---|---:|
| Accuracy | 85.83% |
| Attack Recall | 99.18% |
| FPR | 30.53% |
| FNR | 0.82% |
| ROC-AUC | 0.9847 |
| Average Precision (AP) | 0.9887 |

`average_precision_score` 的結果在本專案中稱為 Average Precision（AP），不直接稱為 PR-AUC。

Confusion Matrix：

```text
              Predicted
              Normal    Attack
Actual Normal  25705     11295
Actual Attack    372     44960
```

TN = 25,705、FP = 11,295、FN = 372、TP = 44,960。

45,332 個 Attack Flow 中成功偵測 44,960 個，因此 Attack Recall 約 99.18%。但 37,000 個 Normal Flow 中有 11,295 個被誤判為 Attack，因此 FPR 30.53% 是目前系統的重要限制。

Classification Report：Normal Precision 0.9857、Recall 0.6947、F1 0.8150；Attack Precision 0.7992、Recall 0.9918、F1 0.8852。

---

## 8. ROC-AUC 與 Average Precision

Final Test ROC-AUC = 0.9847，Average Precision = 0.9887。

ROC-AUC 評估不同 threshold 下模型區分正負樣本的 ranking ability，因此高 ROC-AUC 不代表某個固定 threshold 的 FPR 一定很低。模型在 threshold 0.45 下 FPR = 30.53%，與 ROC-AUC 0.9847 並不矛盾。

---

## 9. Validation vs Test Difference

Validation：Accuracy 95.35%、FPR 11.29%、Recall 98.47%。

Test：Accuracy 85.83%、FPR 30.53%、Recall 99.18%。

Attack Recall 仍維持很高，主要 performance degradation 發生在 Normal Flow 的 False Positive。這可能與 Training/Validation 與官方 Test Data 的 feature distribution 差異或 class overlap 有關，但目前分析不足以直接證明原因是 overfitting。

---

## 10. Error Analysis

Final Test：False Positives = 11,295；False Negatives = 372。

False Positive Attack Probability：Mean 0.6763、Median 0.6595、Min 0.4500、Max 0.9968。代表部分 Normal Flow 並非只是剛越過 0.45，而是模型給出了相對較高的 Attack Probability，因此高 FPR 不能單純歸因於 threshold 從 0.50 降至 0.45。

False Negative Attack Probability：Mean 0.3279、Median 0.3590、Min 0.0020、Max 0.4499。除了接近 threshold 的漏報，也存在被模型高度傾向判斷為 Normal 的 Attack。

---

## 11. FN Rate by Attack Category

| Attack Category | Total Attacks | FN | FN Rate |
|---|---:|---:|---:|
| Fuzzers | 6062 | 308 | 5.0808% |
| Backdoor | 583 | 6 | 1.0292% |
| Shellcode | 378 | 2 | 0.5291% |
| DoS | 4089 | 19 | 0.4647% |
| Analysis | 677 | 2 | 0.2954% |
| Exploits | 11132 | 28 | 0.2515% |
| Generic | 18871 | 6 | 0.0318% |
| Reconnaissance | 3496 | 1 | 0.0286% |
| Worms | 44 | 0 | 0.0000% |

372 個 FN 中有 308 個是 Fuzzers，占約 82.8%。考慮各類別分母後，Fuzzers FN Rate 約 5.08%，也是目前 Testing Set 中主要的漏報來源。

Worms 雖然 FN Rate 為 0%，但只有 44 個 samples，因此不能宣稱模型對 Worms 特別強。

---

## 12. Generated Visualizations

Day 3 產生：

```text
results/confusion_matrix.png
results/roc_curve.png
results/pr_curve.png
results/threshold_tradeoff.png
```

分別呈現 Final Test Confusion Matrix、ROC Curve、Precision-Recall Curve，以及 Validation Threshold Trade-off。Threshold Trade-off 使用 Validation Data，而非 Test Data。

---

## 13. Final Model Persistence

Final Model：

```text
models/xgboost_ids_final.joblib
```

Bundle：

```python
{
    "preprocessor": preprocessor,
    "model": model,
    "feature_columns": X_train.columns.tolist(),
    "threshold": CHOSEN_THRESHOLD
}
```

其中保存 preprocessing、XGBoost model、42 個輸入 features，以及正式 Threshold = 0.45。

---

## 14. External CSV Inference

`predict_csv.py` 已改為載入 `models/xgboost_ids_final.joblib`，並直接從 bundle 讀取：

```python
threshold = ids_pipeline["threshold"]
```

測試：

```bash
python src/predict_csv.py data/sample_input.csv
```

結果：

```text
Classification Threshold : 0.45

Flow #1  0.8939  ATTACK
Flow #2  0.0010  NORMAL
Flow #3  0.9968  ATTACK
Flow #4  0.4755  ATTACK

Total Flows : 4
Normal      : 1
Attack      : 3
```

因此 Final Model → External CSV inference pipeline 已成功打通。

---

## 15. AI-IDS v1.0 Model Freeze

Day 3 完成後正式 Freeze：

```text
Model          : XGBoost
Threshold      : 0.45
Accuracy       : 85.83%
Attack Recall  : 99.18%
FPR            : 30.53%
FNR            : 0.82%
ROC-AUC        : 0.9847
AP             : 0.9887
```

除非發現 implementation bug，v1.0 不再調整 threshold、hyperparameters、模型選擇，也不使用 Test Set 重新調參。

後續集中於 Dashboard、Demo、Documentation、Architecture 與推甄展示。

未來 v2.0 的主要研究方向：

> 在維持高 Attack Recall 的前提下降低 False Positive Rate。

---

## 16. Current System Positioning

目前系統應稱為：

> Machine Learning-Based Network Intrusion Detection Prototype

目前支援符合 UNSW-NB15 42-feature schema 的 Network Flow CSV → ML inference → Attack Probability → NORMAL / ATTACK。

目前尚未實作 Raw Packet Capture → Flow Reconstruction → 42 Feature Extraction，因此不能宣稱為完整 Real-Time IDS。

---


## 17. Day 3 結論

Day 3 完成：

```text
Dataset
   ↓
Train / Validation / Test
   ↓
Preprocessing
   ↓
XGBoost
   ↓
Validation Threshold Selection
   ↓
Final Test Evaluation
   ↓
Error Analysis
   ↓
Model Persistence
   ↓
External CSV Inference
```

AI-IDS v1.0 模型至此 Freeze。

下一階段：**Day 4 — IDS Dashboard**。
