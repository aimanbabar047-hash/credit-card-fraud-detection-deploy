# 💳 Credit Card Fraud Detection

A machine learning project that detects potentially fraudulent credit card transactions using **Logistic Regression** with **SMOTE** for handling class imbalance.

The trained model is deployed as an interactive **Streamlit web application**, allowing users to test sample transactions and enter custom transaction features.

---

## 📌 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions are extremely rare compared to legitimate transactions.

The dataset contains:

- **284,807 transactions**
- **492 fraudulent transactions**
- Fraud rate: approximately **0.17%**

Because of this severe class imbalance, accuracy alone is not a reliable metric for evaluating the model.

This project explores different approaches for handling imbalanced data and deploys the best-performing model as a web application.

---

## 🧠 Machine Learning Approach

The project compares:

1. Baseline Logistic Regression
2. Logistic Regression with SMOTE
3. Class-Weighted Logistic Regression

### Best Model

The deployed model is:

**SMOTE + StandardScaler + Logistic Regression**

SMOTE (Synthetic Minority Over-sampling Technique) was used to increase representation of the minority fraud class in the training data.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 99.91% | 82.67% | 63.27% | 71.68% | 0.9605 |
| SMOTE Logistic Regression | 98.99% | 13.41% | 89.80% | 23.34% | **0.9765** |
| Class-Weighted Logistic Regression | 97.55% | 6.10% | 91.84% | 11.44% | 0.9721 |

The SMOTE model achieved the highest **ROC-AUC of 0.9765** while substantially improving fraud recall compared with the baseline model.

---

## 🌐 Streamlit Web Application

The model has been deployed as an interactive Streamlit application.

### Features

- 🧪 Test predefined legitimate transactions
- 🚨 Test predefined fraudulent transactions
- ✍️ Enter custom transaction features
- 📊 View fraud probability
- 🔍 View model prediction

### 🌐 Live Application

🚀 **[Launch Credit Card Fraud Detection App](https://credit-card-fraud-detection-deploy-mmeh7v94fjh4uur4qmxuae.streamlit.app/)**

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Logistic Regression
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

---

## 📂 Project Structure

```text
credit-card-fraud-detection-deploy/
│
├── app.py
├── app - backupCopy.py
├── Credit_Card_Fraud_Deployment.ipynb
├── feature_names.pkl
├── smote_fraud_detection_model.pkl
├── requirements.txt
└── README.md
