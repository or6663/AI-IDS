# Day 02 — XGBoost、模型評估與 IDS Inference Pipeline

## 今日目標

Day 2 的主要目標：

1. 建立 Git / GitHub Repository
2. 建立 XGBoost IDS Model
3. 比較 Logistic Regression、Random Forest、XGBoost
4. 加入 IDS 相關 Evaluation Metrics
5. 分析 Classification Threshold
6. 保存訓練完成的 Model
7. 建立 Inference Pipeline
8. 讓 IDS 可以讀取外部 CSV 進行預測

---

# 1. Git 與 GitHub

本專案開始使用 Git 進行版本控制，並建立 GitHub Repository。

Repository：

```text
or6663/AI-IDS
```

本機 Branch：

```text
main
```

Git 的基本開發流程：

```text
修改程式
    ↓
git status
    ↓
git add
    ↓
git commit
    ↓
git push
    ↓
GitHub
```

其中：

- `git add`：將修改加入 Staging Area
- `git commit`：建立一個版本紀錄
- `git push`：將本機 Commit 上傳到 GitHub

---

# 2. .gitignore

專案使用 `.gitignore` 排除不適合上傳 GitHub 的檔案。

例如：

```text
.venv/
data/*.csv
models/*.joblib
models/*.pkl
.env
```

原因：

### `.venv`

Python Virtual Environment 可以透過：

```bash
pip install -r requirements.txt
```

重新建立，不需要上傳。

### Dataset

Dataset 通常容量較大，而且可以由原始來源取得，因此不直接加入 Repository。

### Trained Model

目前 XGBoost Model 產生：

```text
models/xgboost_ids.joblib
```

模型屬於 Binary Artifact，因此目前不加入 Git Repository。

---

# 3. XGBoost

今天新增第三個 Machine Learning Model：

```text
XGBoost
```

版本：

```text
xgboost==3.4.1
```

XGBoost 屬於 Gradient Boosting Tree Model。

---

# 4. Random Forest vs XGBoost

Random Forest 與 XGBoost 都屬於 Tree-based Ensemble Learning，但核心概念不同。

## Random Forest

主要概念：

```text
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤
...     ├──→ 綜合預測
Tree N ─┘
```

多棵 Decision Trees 相對獨立地建立，再整合結果。

主要屬於：

```text
Bagging
```

---

## XGBoost

XGBoost 使用 Boosting：

```text
Tree 1
   ↓
分析目前模型的錯誤
   ↓
Tree 2 改善前面的錯誤
   ↓
Tree 3 繼續改善
   ↓
...
   ↓
Final Prediction
```

主要屬於：

```text
Boosting
```

---

# 5. XGBoost Preprocessing

原始 Model Input：

```text
42 Features
```

其中：

```text
3 Categorical Features
39 Numerical Features
```

Categorical：

```text
proto
service
state
```

使用：

```python
OneHotEncoder(handle_unknown="ignore")
```

Numerical Features：

```text
passthrough
```

XGBoost 是 Tree-based Model，因此 Numerical Features 不需要 StandardScaler。

處理完成後：

```text
Training:
(175341, 194)

Testing:
(82332, 194)
```

---

# 6. XGBoost Initial Configuration

目前 XGBoost 使用：

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

重要參數：

### n_estimators

```text
100
```

表示使用 100 個 boosting stages / trees。

### max_depth

```text
6
```

限制 Tree 最大深度。

Tree 越深通常可以學習更複雜的 Feature Interaction，但也可能增加 Overfitting 風險。

### learning_rate

```text
0.1
```

控制每一輪新增 Tree 對整體 Model 的修正幅度。

### objective

```text
binary:logistic
```

因為目前 IDS 任務為：

```text
0 = Normal
1 = Attack
```

屬於 Binary Classification。

---

# 7. 三模型比較

目前測試結果：

| Model | Accuracy | FP | FN |
|---|---:|---:|---:|
| Logistic Regression | 80.97% | 14,419 | 1,245 |
| Random Forest | 87.10% | 9,963 | 659 |
| XGBoost | 87.26% | 9,896 | 592 |

在目前的初始設定下，XGBoost 測試結果略高於 Random Forest。

但差距只有：

```text
87.26% - 87.10%
= 0.16 percentage points
```

因此不能僅根據這次結果宣稱：

```text
XGBoost 一定優於 Random Forest
```

比較合理的說法：

> 在目前資料前處理與初始模型設定下，XGBoost 在 Test Set 上取得略高於 Random Forest 的結果。

---

# 8. XGBoost Confusion Matrix

XGBoost：

```text
[[27104  9896]
 [  592 44740]]
```

表示：

```text
TN = 27104
FP = 9896
FN = 592
TP = 44740
```

表格：

| | Predicted Normal | Predicted Attack |
|---|---:|---:|
| Actual Normal | TN = 27,104 | FP = 9,896 |
| Actual Attack | FN = 592 | TP = 44,740 |

---

# 9. False Positive 與 False Negative

## False Positive（FP）

```text
實際：Normal
預測：Attack
```

也就是：

```text
False Alarm
```

目前：

```text
FP = 9,896
```

---

## False Negative（FN）

```text
實際：Attack
預測：Normal
```

代表 IDS 漏掉真正的攻擊。

目前：

```text
FN = 592
```

在 Network Security 情境中，FN 特別重要，因為代表真正 Attack 沒有被 IDS 偵測到。

但 FP 過多同樣會造成大量 False Alarm，增加管理負擔。

---

# 10. FPR

False Positive Rate：

```text
FPR = FP / (FP + TN)
```

目前：

```text
FPR = 0.2675
    = 26.75%
```

意思：

> 在目前 Test Set 的真正 Normal Samples 中，約 26.75% 被模型錯誤判斷為 Attack。

不能直接將這個比例宣稱為真實世界部署後的 False Alarm Rate，因為實際網路環境的資料分布可能與 UNSW-NB15 Test Set 不同。

---

# 11. FNR

False Negative Rate：

```text
FNR = FN / (FN + TP)
```

目前：

```text
FNR = 0.0131
    = 1.31%
```

意思：

> 在目前 Test Set 的真正 Attack Samples 中，約 1.31% 被錯誤判斷為 Normal。

Attack Recall 與 FNR 大致互補：

```text
Attack Recall ≈ 98.69%
FNR           ≈  1.31%
```

---

# 12. ROC-AUC

目前：

```text
ROC-AUC = 0.9845
```

ROC-AUC 用來衡量 Model 在不同 Classification Threshold 下區分 Positive / Negative Samples 的能力。

ROC-AUC 並不等於 Accuracy。

Accuracy 是特定分類決策下：

```text
預測正確數 / 全部 Samples
```

ROC-AUC 則著重於模型跨不同 Threshold 的排序與區分能力。

因此：

```text
Accuracy = 0.8726
ROC-AUC  = 0.9845
```

並不矛盾。

---

# 13. PR-AUC

目前：

```text
PR-AUC = 0.9886
```

PR：

```text
Precision - Recall
```

PR-AUC 用來觀察不同 Threshold 下 Precision 與 Recall 的整體關係。

在 IDS、Anomaly Detection，以及類別不平衡問題中，是值得搭配 ROC-AUC 觀察的指標。

---

# 14. Classification Threshold

XGBoost 使用：

```python
model.predict_proba(...)
```

可以取得：

```text
Attack Probability
```

例如：

```text
Attack Probability = 0.8830
```

使用 Threshold：

```text
0.5
```

則：

```text
Probability >= 0.5
→ ATTACK

Probability < 0.5
→ NORMAL
```

---

# 15. Threshold Experiment

測試：

```text
0.3
0.4
0.5
0.6
0.7
0.8
0.9
```

結果：

| Threshold | Accuracy | FPR | FNR | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 81.61% | 40.71% | 0.18% | 15,061 | 81 |
| 0.40 | 83.86% | 35.25% | 0.54% | 13,043 | 244 |
| 0.50 | 87.26% | 26.75% | 1.31% | 9,896 | 592 |
| 0.60 | 90.19% | 18.74% | 2.52% | 6,934 | 1,143 |
| 0.70 | 91.58% | 13.46% | 4.30% | 4,981 | 1,948 |
| 0.80 | 92.66% | 7.80% | 6.97% | 2,885 | 3,158 |
| 0.90 | 92.89% | 2.35% | 10.99% | 871 | 4,980 |

觀察：

```text
Threshold ↑
     ↓
判斷 Attack 的條件變嚴格
     ↓
FP ↓
但
FN ↑
```

---

# 16. 為什麼不能直接選 Threshold = 0.9？

因為：

```text
Threshold 0.9
Accuracy = 92.89%
```

雖然是目前測試中最高 Accuracy，但：

```text
FNR = 10.99%
FN  = 4,980
```

代表漏掉的 Attack 大幅增加。

IDS 不應只根據 Accuracy 選 Threshold。

需要根據：

```text
False Positive Cost
vs
False Negative Cost
```

決定適合的 Operating Point。

---

# 17. 為什麼不能直接使用 Test Set 選 Threshold？

目前 Threshold Experiment 使用 Test Set 是為了分析模型行為。

但如果直接觀察 Test Set：

```text
哪個 Threshold 表現最好？
```

然後選擇該 Threshold 作為正式系統設定，就等於使用 Test Set 參與模型決策。

較正式的流程：

```text
Training Data
      │
      ├── Training Subset
      │        ↓
      │      Train
      │
      └── Validation Subset
               ↓
         Select Threshold
               ↓
          Threshold 固定
               ↓
           Test Set
               ↓
        Final Evaluation
```

Test Set 應盡可能保留作為最後的獨立評估資料。

---

# 18. Training vs Inference

## Training

```text
Training Dataset
       ↓
Preprocessing
       ↓
model.fit()
       ↓
Trained Model
```

Training 的目的是讓模型從資料中學習。

---

## Inference

```text
New Data
    ↓
Preprocessing
    ↓
model.predict()
或
model.predict_proba()
    ↓
Prediction
```

Inference 使用已經訓練完成的模型。

不需要每收到一筆 Network Flow 就重新 Training。

---

# 19. Model Persistence

使用：

```python
joblib
```

保存：

```text
Preprocessor
Model
Feature Columns
```

檔案：

```text
models/xgboost_ids.joblib
```

內容：

```python
{
    "preprocessor": preprocessor,
    "model": model,
    "feature_columns": X_train.columns.tolist()
}
```

---

# 20. 為什麼 Preprocessor 也必須保存？

Training 時：

```text
42 Features
    ↓
已 Fit 的 One-Hot Encoder
    ↓
194 Features
    ↓
XGBoost
```

Inference 時，新資料必須使用**相同的 Feature Transformation**。

如果重新建立並 Fit 一個新的 Encoder：

```text
Category Mapping
Feature Order
Feature Dimensions
```

可能與 Training 時不同。

因此必須保存並重用 Training 階段已 Fit 的 Preprocessor。

---

# 21. 單筆 Inference

建立：

```text
src/predict.py
```

載入：

```text
models/xgboost_ids.joblib
```

後直接對 Network Flow 進行 Prediction，而不重新 Training。

測試找到四種案例：

| Case | Sample | Attack Probability | Prediction | Truth |
|---|---:|---:|---|---|
| TN | 8 | 0.0007 | Normal | Normal |
| FP | 0 | 0.8830 | Attack | Normal |
| FN | 254 | 0.4307 | Normal | Attack |
| TP | 243 | 0.9967 | Attack | Attack |

其中：

### False Positive Example

```text
Sample 0

Attack Probability = 0.8830
Prediction = ATTACK
Truth = NORMAL
```

代表模型非常有信心地產生一次 False Alarm。

### False Negative Example

```text
Sample 254

Attack Probability = 0.4307
Threshold = 0.5

Prediction = NORMAL
Truth = ATTACK
Attack Category = Exploits
```

如果 Threshold 改為：

```text
0.4
```

則：

```text
0.4307 >= 0.4
```

這筆會變成 Attack。

這個案例可以直接說明 Threshold 對 FN / FP 的影響。

---

# 22. External CSV Inference

建立：

```text
src/predict_csv.py
```

目前可以執行：

```bash
python src/predict_csv.py data/sample_input.csv
```

輸入 CSV：

```text
4 Samples
42 Features
```

而且不包含：

```text
label
attack_cat
```

輸出：

```text
Flow #1
Attack Probability : 0.8830
Prediction         : ATTACK

Flow #2
Attack Probability : 0.0007
Prediction         : NORMAL

Flow #3
Attack Probability : 0.9967
Prediction         : ATTACK

Flow #4
Attack Probability : 0.4307
Prediction         : NORMAL
```

Summary：

```text
Total Flows : 4
Normal      : 2
Attack      : 2
```

因此目前已經可以將：

```text
Training
```

與：

```text
Inference
```

分離。

---

# 23. 目前系統的定位

目前可以描述為：

> 基於 Machine Learning 的 Network Intrusion Detection Prototype，可對符合 UNSW-NB15 Feature Schema 的 Network Flow CSV 進行 Normal / Attack 二元分類。

目前不能描述成：

> Real-time Network IDS

因為尚未完成：

```text
Network Interface
       ↓
Packet Capture
       ↓
Flow Reconstruction
       ↓
Feature Extraction
       ↓
42 UNSW-NB15 Features
       ↓
AI Model
```

目前主要完成的是：

```text
42 Features
      ↓
AI Model
      ↓
Prediction
```

---

# 24. 今日新增檔案

```text
src/train_xgboost.py
src/predict.py
src/predict_csv.py
src/create_sample_input.py
docs/day02.md
```

另外更新：

```text
README.md
requirements.txt
```

本機產生但不 Push：

```text
models/xgboost_ids.joblib
data/sample_input.csv
```

---

# 25. 教授可能提問

## Q1：為什麼除了 Random Forest 還要使用 XGBoost？

答：

Random Forest 與 XGBoost 雖然都是 Tree-based Ensemble Models，但 Random Forest 主要使用 Bagging，而 XGBoost 使用 Boosting。

因此本專案希望比較不同 Ensemble Learning Strategy 在 Network Intrusion Detection 上的表現。

---

## Q2：為什麼 XGBoost 不使用 StandardScaler？

答：

XGBoost 主要使用 Decision Tree Split 進行分類，Tree-based Model 主要根據 Feature Value 的排序與 Threshold 進行分裂，因此不像 Logistic Regression 對 Feature Scale 那麼敏感。

所以目前 Numerical Features 採用 passthrough。

---

## Q3：為什麼 Accuracy 87.26%，ROC-AUC 卻有 98.45%？

答：

Accuracy 是特定 Classification Decision 下的正確比例，而 ROC-AUC 衡量不同 Threshold 下模型區分 Positive 與 Negative Samples 的整體能力。

因此兩個 Metric 衡量的內容不同，不需要具有相同數值。

---

## Q4：IDS 最重要的是 Accuracy 嗎？

答：

不是。

IDS 還需要考慮：

```text
False Positive
False Negative
Precision
Recall
FPR
FNR
```

例如 False Negative 代表真正的 Attack 被模型判定為 Normal；False Positive 過高則會造成大量 False Alarm。

---

## Q5：為什麼不直接使用 Threshold 0.9？

答：

雖然目前 Test Set 中 Threshold 0.9 的 Accuracy 較高，而且 FPR 降低，但 FNR 增加到 10.99%。

代表大量真正的 Attack 會被判斷成 Normal。

因此 Threshold 應根據 IDS 對 False Positive 與 False Negative 的需求進行選擇，而不能只看 Accuracy。

---

## Q6：為什麼不能使用 Test Set 選最佳 Threshold？

答：

如果使用 Test Set 的結果決定 Threshold，就代表 Test Set 參與模型設定。

較正式的方法應使用 Validation Set 選擇 Threshold，再使用獨立 Test Set 進行最終評估。

---

## Q7：Training 與 Inference 有什麼不同？

答：

Training 使用：

```python
model.fit()
```

讓模型從 Training Data 學習。

Inference 則載入已經訓練完成的模型，使用：

```python
model.predict()
```

或：

```python
model.predict_proba()
```

對新資料進行預測，不需要重新 Training。

---

## Q8：為什麼保存 Model 還要保存 Preprocessor？

答：

因為新資料必須經過與 Training 時完全相同的 Feature Transformation。

例如 One-Hot Encoding 的 Category Mapping、Feature Order 與輸出維度都必須保持一致。

因此需要保存 Training 時已經 Fit 完成的 Preprocessor。

---

## Q9：你的 IDS 現在可以即時監控網路嗎？

答：

目前還不行。

目前系統已經完成 Machine Learning Detection 與 CSV-based Inference，但尚未完成 Packet Capture、Flow Reconstruction 與 Real-time Feature Extraction。

因此目前定位為 Network Intrusion Detection Prototype，而不是完整的 Real-time IDS。

---

## Q10：目前模型最大的問題是什麼？

答：

目前 XGBoost 的 Attack Recall 很高，FNR 約為 1.31%，但是在 Threshold 0.5 下 FPR 約為 26.75%。

因此目前的重要問題之一是降低 False Positive，同時避免造成過多 False Negative。

後續可以透過 Validation-based Threshold Selection、Error Analysis、Feature Analysis 或模型調整進一步研究。

---

# 26. Day 2 結論

Day 2 完成第三個模型 XGBoost，並與 Logistic Regression、Random Forest 進行比較。

目前 XGBoost：

```text
Accuracy = 87.26%
ROC-AUC  = 98.45%
PR-AUC   = 98.86%

FPR = 26.75%
FNR = 1.31%
```

此外完成 Classification Threshold Experiment，觀察到：

```text
Threshold ↑
→ False Positive ↓
→ False Negative ↑
```

最後將 XGBoost Model、Preprocessor 與 Feature Columns 保存，並建立 CSV-based Inference Pipeline。

因此專案已從：

```text
Dataset
↓
Training
↓
Evaluation
```

進一步發展為：

```text
Training
↓
Save Model
↓
Load Model
↓
New Input
↓
Inference
↓
Attack Probability
↓
Normal / Attack
```

---

# Day 3 預計方向

下一階段優先考慮：

1. Validation Set 與 Threshold Selection
2. ROC Curve / Precision-Recall Curve
3. XGBoost Explainability / Feature Analysis
4. False Positive / False Negative Error Analysis
5. 改善 Prediction Output
6. 開始規劃 IDS Demo Interface

核心目標不是無限制增加模型，而是逐步讓目前 Prototype：

```text
可評估
→ 可解釋
→ 可推論
→ 可展示
```