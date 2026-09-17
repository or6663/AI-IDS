# AI-IDS Day 01 — Binary IDS Baseline

## 1. 今日目標

建立 AI-IDS 專案的第一版 Binary Intrusion Detection System（Binary IDS），使用 UNSW-NB15 Dataset，將網路流量分類為：

- `0`：Normal
- `1`：Attack

Day 1 的重點不是追求最高準確率，而是先建立完整且可比較的 Machine Learning Baseline Pipeline：

Dataset → Preprocessing → Model Training → Prediction → Evaluation → Model Comparison

---

# 2. Dataset

使用資料集：

**UNSW-NB15**

目前使用：

- Training Set：175,341 筆
- Testing Set：82,332 筆
- 原始欄位數：45

Training Label Distribution：

- Attack (`1`)：119,341
- Normal (`0`)：56,000

---

# 3. Feature / Label 分離

模型預測目標：

`label`

移除三個欄位：

- `id`
- `attack_cat`
- `label`

因此模型實際使用：

**42 Features**

其中：

- Categorical Features：3
- Numerical Features：39

Categorical Features：

- `proto`
- `service`
- `state`

---

# 4. 為什麼移除 attack_cat？

`attack_cat` 代表該筆資料的攻擊種類。

例如一筆資料如果已經標示為某種 Attack，再利用 `attack_cat` 預測它是不是 Attack，相當於提前把答案相關資訊提供給模型。

這會造成：

**Data Leakage（資料洩漏）**

因此 Binary IDS 不將 `attack_cat` 作為輸入 Feature。

---

# 5. Data Preprocessing

## 5.1 Categorical Features

三個類別型 Feature：

`proto`, `service`, `state`

使用：

**One-Hot Encoding**

將類別資料轉換成 Machine Learning Model 可以處理的數值形式。

例如：

`proto = tcp / udp / icmp`

可能展開成：

`proto_tcp`, `proto_udp`, `proto_icmp`

因此原本的 42 Features 經過 One-Hot Encoding 後變成：

**194 Features**

這是正常現象。

---

## 5.2 Numerical Features

Logistic Regression 使用：

**StandardScaler**

對 Numerical Features 進行標準化。

Random Forest 則不使用 StandardScaler，Numerical Features 直接 passthrough。

原因是 Decision Tree / Random Forest 主要根據 Feature 的切分點進行判斷，通常不受不同 Feature 數值尺度影響。

---

# 6. 避免 Preprocessing Data Leakage

Training Data：

`preprocessor.fit_transform(X_train)`

Testing Data：

`preprocessor.transform(X_test)`

Testing Data 不可以重新 `fit_transform()`。

原因是模型建立前處理規則時只能學習 Training Set 的資訊。

如果使用 Test Set 重新 Fit，等於在模型評估前使用了測試資料資訊，可能造成 Data Leakage。

---

# 7. 第一個模型：Logistic Regression

Logistic Regression 作為第一個 Baseline Model。

Baseline 的用途：

先建立一個相對簡單的基準模型，後續使用 Random Forest、XGBoost 等模型時，才能判斷新模型是否真的帶來改善。

## Logistic Regression Result

Accuracy：

**80.97%**

Classification Report：

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Normal | 0.95 | 0.61 | 0.74 |
| Attack | 0.75 | 0.97 | 0.85 |

Confusion Matrix：

| Actual / Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 22,581 | 14,419 |
| Attack | 1,245 | 44,087 |

因此：

- TN = 22,581
- FP = 14,419
- FN = 1,245
- TP = 44,087

## Logistic Regression 分析

優點：

Attack Recall 約為 97%，代表絕大部分真正的 Attack 都能被偵測。

缺點：

False Positive 高達 14,419。

大量正常流量被誤判為 Attack，因此 False Alarm 問題明顯。

結論：

**Logistic Regression 能有效抓到攻擊，但誤報率偏高。**

---

# 8. Confusion Matrix

Binary Classification 中：

## True Positive（TP）

實際：Attack  
預測：Attack

成功偵測攻擊。

## True Negative（TN）

實際：Normal  
預測：Normal

正確判斷正常流量。

## False Positive（FP）

實際：Normal  
預測：Attack

正常流量被誤判成攻擊。

在 IDS 中會造成：

**False Alarm（誤報）**

## False Negative（FN）

實際：Attack  
預測：Normal

真正的攻擊被模型當成正常流量。

這代表：

**攻擊成功逃過 IDS。**

因此在 IDS 中，False Negative 是非常重要的評估指標之一。

---

# 9. 第二個模型：Random Forest

Random Forest 使用：

**100 Decision Trees**

設定：

`n_estimators = 100`

並使用：

`n_jobs = -1`

讓模型使用可用的 CPU Core 進行平行運算。

Random Forest 不使用 StandardScaler。

---

# 10. Random Forest Result

Accuracy：

**87.10%**

Classification Report：

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Normal | 0.98 | 0.73 | 0.84 |
| Attack | 0.82 | 0.99 | 0.89 |

Confusion Matrix：

| Actual / Predicted | Normal | Attack |
|---|---:|---:|
| Normal | 27,037 | 9,963 |
| Attack | 659 | 44,673 |

因此：

- TN = 27,037
- FP = 9,963
- FN = 659
- TP = 44,673

Attack Recall 約：

**98.55%**

---

# 11. Logistic Regression vs Random Forest

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.97% | **87.10%** |
| Normal Precision | 0.95 | **0.98** |
| Normal Recall | 0.61 | **0.73** |
| Normal F1 | 0.74 | **0.84** |
| Attack Precision | 0.75 | **0.82** |
| Attack Recall | 0.97 | **0.99** |
| Attack F1 | 0.85 | **0.89** |
| False Positive | 14,419 | **9,963** |
| False Negative | 1,245 | **659** |

Random Forest 相較 Logistic Regression：

- Accuracy：+6.13 percentage points
- False Positive：減少 4,456
- False Negative：減少 586
- Attack Recall 提升
- Normal Recall 提升

因此目前：

**Random Forest 為 Day 1 表現最佳模型。**

---

# 12. 為什麼 Random Forest 表現可能較好？

Logistic Regression 建立的分類關係相對簡單。

Network Intrusion Detection 則可能涉及多個 Features 之間複雜、非線性的關係。

例如：

Protocol + TTL + Packet Rate + Connection State + Traffic Load

可能共同構成某種攻擊行為。

Random Forest 由大量 Decision Trees 組成，因此可以學習較複雜的非線性 Feature Relationships。

Day 1 實驗結果顯示 Random Forest 在目前資料上明顯優於 Logistic Regression。

---

# 13. Random Forest Feature Importance

Top 15：

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
| 11 | tcprtt | 0.030943 |
| 12 | ct_srv_dst | 0.029641 |
| 13 | sbytes | 0.029139 |
| 14 | smean | 0.025880 |
| 15 | dbytes | 0.023550 |

目前模型較依賴：

- TTL
- Network Load
- Packet Rate
- Packet Timing
- Bytes
- Connection Statistics

等 Network Traffic Features。

注意：

**Feature Importance 不代表因果關係。**

例如：

`ct_state_ttl importance = 0.108`

只能說：

> 在目前訓練出的 Random Forest Model 中，ct_state_ttl 對分類決策具有較高的重要性。

不能說：

> ct_state_ttl 造成 Network Attack。

---

# 14. Day 1 完成的程式

目前主要程式：

`src/explore_data.py`

用途：
探索 UNSW-NB15 Dataset。

`src/preprocess.py`

用途：
確認 Features、Labels、Categorical / Numerical Features。

`src/train_logistic.py`

用途：
建立 Logistic Regression Binary IDS Baseline。

`src/train_random_forest.py`

用途：
建立 Random Forest Binary IDS，並分析 Feature Importance。


---

# 16. Day 1 結論

Day 1 已成功建立完整 Binary IDS Machine Learning Pipeline：

UNSW-NB15 Dataset  
↓  
Feature / Label Separation  
↓  
Data Preprocessing  
↓  
One-Hot Encoding  
↓  
Logistic Regression Baseline  
↓  
Random Forest  
↓  
Classification Report  
↓  
Confusion Matrix  
↓  
Feature Importance

目前最佳模型：

**Random Forest**

Accuracy：

**87.10%**

Attack Recall：

**約 98.55%**

False Negative：

**659**

目前主要問題：

仍有 **9,963 個 Normal Samples 被誤判為 Attack**，False Positive 仍偏高。

後續工作將繼續改善模型表現、分析模型，以及擴展 IDS 功能。

---

# 17. Day 2 預定工作

1. 建立第三個模型（XGBoost）
2. 與 Logistic Regression / Random Forest 比較
3. 建立統一 Model Evaluation / Comparison
4. 分析 False Positive / False Negative
5. 視進度開始模型儲存與實際 IDS Prediction Pipeline

Day 1 Status：

**Completed**