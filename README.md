# 🏦 Loan Status Prediction using Machine Learning

Predict whether a loan application will be approved based on applicant features using various classification algorithms.

---

## 📁 Project Overview

This project builds a supervised machine learning pipeline that uses applicant financial and personal data to classify whether a loan should be approved (`Loan_Status`).

We applied preprocessing steps and trained multiple models including Logistic Regression, Decision Tree, Random Forest, and XGBoost.

---

## 🧠 Problem Statement

Manually reviewing loan applications can be slow and inconsistent. The goal is to automate this process by learning from past approved and rejected loans, helping financial institutions make faster, data-driven decisions.

---

## 📊 Dataset Description

The dataset includes the following key features:

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (Target)

> ℹ️ Replace `loan_data.csv` with your actual dataset if needed.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**, **Seaborn**

---

## ⚙️ ML Pipeline

### 1. Data Preprocessing
- Handled missing values using `SimpleImputer`
- Label encoded categorical variables
- Feature scaling with `StandardScaler`

### 2. Model Training
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost Classifier

### 3. Evaluation Metrics
- Accuracy
- Precision
- Recall
- Classification Report

---

## 📈 Model Results

| Model               | Accuracy | Precision | Recall |
|--------------------|----------|-----------|--------|
| Logistic Regression| ~82%     | ~80%      | ~78%   |
| Decision Tree      | ~78%     | ~76%      | ~75%   |
| Random Forest      | ~84%     | ~82%      | ~80%   |
| XGBoost            | ~85%     | ~83%      | ~81%   |

✅ **XGBoost** delivered the best performance across all metrics.

---

## 🧪 How to Run This Project

### 1. Clone this repository:
```bash
git clone https://github.com/Surya0400/loan-status-prediction.git
cd loan-status-prediction

pip install -r requirements.txt

python loan_status_prediction.py

---
loan-status-prediction/
│
├── loan_status_prediction.py   # Main code
├── loan_data.csv               # Dataset
├── requirements.txt            # Required Python packages
└── README.md                   # Project description

