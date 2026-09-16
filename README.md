# AI-Based Network Intrusion Detection System

A machine learning-based Network Intrusion Detection System (IDS) built using the UNSW-NB15 dataset.

The current system performs **binary classification** of network traffic:

- `0` — Normal traffic
- `1` — Attack traffic

This project explores different machine learning approaches for detecting malicious network traffic and compares their detection performance.

---

## Project Status

🚧 **Work in Progress**

Currently implemented:

- Dataset exploration
- Data preprocessing
- Categorical feature encoding
- Logistic Regression baseline
- Random Forest classifier
- Confusion matrix evaluation
- Feature importance analysis

---

## Dataset

This project uses the **UNSW-NB15** network intrusion detection dataset.

Current dataset split:

| Dataset | Samples |
|---|---:|
| Training Set | 175,341 |
| Testing Set | 82,332 |

The original dataset contains 45 columns.

For binary classification, the following columns are excluded from model input:

- `id`
- `attack_cat`
- `label`

This results in **42 input features**:

- 3 categorical features
- 39 numerical features

The categorical features are:

- `proto`
- `service`
- `state`

After One-Hot Encoding, the current feature space contains **194 features**.

> The dataset files are not included in this repository.

---

## Data Preprocessing

Categorical features are processed using **One-Hot Encoding**.

For Logistic Regression, numerical features are standardized using **StandardScaler**.

For Random Forest, numerical features are used directly because tree-based models generally do not require feature scaling.

The preprocessing pipeline is fitted only on the training set to prevent data leakage.

---

## Models

### Logistic Regression

Logistic Regression is used as the initial baseline model.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 80.97% |
| Attack Precision | 0.75 |
| Attack Recall | 0.97 |
| Attack F1-score | 0.85 |
| False Positives | 14,419 |
| False Negatives | 1,245 |

The model achieves high attack recall but produces a relatively large number of false positives.

### Random Forest

The second model uses a Random Forest classifier with 100 decision trees.

Results:

| Metric | Result |
|---|---:|
| Accuracy | **87.10%** |
| Attack Precision | **0.82** |
| Attack Recall | **0.99** |
| Attack F1-score | **0.89** |
| False Positives | **9,963** |
| False Negatives | **659** |

Random Forest improves both overall classification performance and attack detection compared with the Logistic Regression baseline.

---

## Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.97% | **87.10%** |
| Normal Recall | 0.61 | **0.73** |
| Attack Precision | 0.75 | **0.82** |
| Attack Recall | 0.97 | **0.99** |
| Attack F1-score | 0.85 | **0.89** |
| False Positives | 14,419 | **9,963** |
| False Negatives | 1,245 | **659** |

At the current stage, Random Forest provides the strongest performance among the evaluated models.

---

## Random Forest Feature Importance

The most important features identified by the current Random Forest model include:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `ct_state_ttl` | 0.1081 |
| 2 | `sttl` | 0.0957 |
| 3 | `ackdat` | 0.0527 |
| 4 | `sload` | 0.0498 |
| 5 | `dload` | 0.0488 |
| 6 | `dttl` | 0.0449 |
| 7 | `rate` | 0.0436 |
| 8 | `dinpkt` | 0.0411 |
| 9 | `dmean` | 0.0349 |
| 10 | `sinpkt` | 0.0345 |

Feature importance represents the relative contribution of features to the Random Forest classification process and does not imply causality.

---

## Project Structure

```text
AI-IDS/
│
├── data/
│   └── UNSW-NB15 dataset (not tracked by Git)
│
├── docs/
│   └── day01.md
│
├── models/
│
├── results/
│
├── src/
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── train_logistic.py
│   └── train_random_forest.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository and create a Python virtual environment.

Install the required packages:

```bash
pip install -r requirements.txt
```

The project currently uses:

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

---

## Current Research Questions

The project currently focuses on the following questions:

1. How effectively can machine learning distinguish normal and malicious network traffic?
2. How do different machine learning models compare for binary intrusion detection?
3. How can false positives and false negatives be reduced?
4. Which network traffic features contribute most to attack detection?

---

## Future Work

Planned development includes:

- Additional machine learning models
- Improved model evaluation
- False positive / false negative analysis
- Model persistence
- Prediction pipeline for new network traffic
- Visualization of model performance
- Prototype IDS interface
- Model explainability analysis

---

## Development Notes

Detailed development and experiment notes are available in the `docs/` directory.

The project is currently under active development.