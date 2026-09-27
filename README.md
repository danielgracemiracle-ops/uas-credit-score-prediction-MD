# 💳 Credit Score Prediction

A **Machine Learning classification project** designed to predict a customer's credit score category based on financial and credit-related information.

The project covers the complete machine learning workflow, from **data preprocessing and feature engineering to model training, evaluation, and prediction**.

---

## 📌 Project Overview

Credit scoring is an important process in financial services for assessing a customer's creditworthiness.

This project uses customer financial information to build a machine learning classification model capable of predicting credit score categories.

The main workflow includes:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Data Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Prediction
```

---

## 🎯 Objectives

The objectives of this project are:

1. Analyze financial and credit-related customer data.
2. Perform data cleaning and preprocessing.
3. Engineer relevant features for credit scoring.
4. Train multiple machine learning classification models.
5. Compare model performance using classification metrics.
6. Select the most suitable model based on evaluation results.
7. Build a reproducible machine learning pipeline.

---

## 📊 Dataset

The dataset contains customer financial and credit-related information.

The original dataset consists of:

* **25,000 records**
* **29 features**

After preprocessing and feature engineering, the final dataset contains **25 features**.

One of the engineered features is:

* `Credit_History_Months`
* `Debt_to_Income_Ratio`

These features were created to provide additional information about the customer's credit history and financial condition.

---

## 🔧 Data Preprocessing

Several preprocessing steps were performed before model training:

* Handling missing values
* Handling inconsistent data
* Converting categorical variables
* Feature engineering
* Removing unnecessary columns
* Encoding categorical features
* Preparing numerical features
* Preparing the final training dataset

### Feature Engineering

Two important engineered features include:

#### Credit History Months

Represents the customer's credit history duration in months.

#### Debt-to-Income Ratio

Measures the relationship between a customer's debt and income.

```text
Debt-to-Income Ratio =
Total Debt / Monthly Income
```

---

## 🤖 Machine Learning Models

Two classification algorithms were evaluated:

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a more robust classification model.

### XGBoost

XGBoost is a gradient boosting algorithm that builds an ensemble of decision trees sequentially to improve predictive performance.

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### Results

| Model         |   Accuracy |   F1-Score |
| ------------- | ---------: | ---------: |
| Random Forest | **0.7496** | **0.7494** |
| XGBoost       |     0.7264 |          — |

The Random Forest model achieved the highest accuracy in the final evaluation.

> The reported results are based on the final script evaluation.

---

## 🔍 Feature Importance

Feature importance analysis was performed to understand which variables contributed most to the Random Forest model.

The top features included:

| Feature              | Importance |
| -------------------- | ---------: |
| Outstanding Debt     |     0.0960 |
| Interest Rate        |     0.0719 |
| Delay from Due Date  |     0.0587 |
| Changed Credit Limit |     0.0539 |
| Credit Mix           |     0.0529 |

These values represent the model's feature importance and should not be interpreted as causal relationships.

---

## 🏗️ Project Architecture

```text
                    ┌─────────────────┐
                    │   Raw Dataset   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Ingestion     │
                    │    ingest.py    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Preprocessing   │
                    │ preprocessing.py│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model Training  │
                    │   training.py   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Evaluation    │
                    │  evaluation.py  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Prediction   │
                    │     main.py     │
                    └─────────────────┘
```

---

## 📂 Project Structure

```text
Credit-Score-Prediction/
│
├── ai_env/
│
├── ingest.py
├── preprocessing.py
├── training.py
├── evaluation.py
├── main.py
│
├── requirements.txt
├── README.md
│
└── mlruns/
    └── CreditScoreExperiment/
```

---

## 📊 Experiment Tracking

The project uses **MLflow** to track machine learning experiments.

Experiment:

```text
CreditScoreExperiment
```

Tracking URI:

```text
file:./mlruns
```

MLflow was used to organize and track model experiments, parameters, and evaluation results.

---

## 🧪 Machine Learning Pipeline

The project follows a modular pipeline:

### 1. Data Ingestion

`ingest.py`

Responsible for loading and preparing the raw dataset.

### 2. Preprocessing

`preprocessing.py`

Handles:

* Data cleaning
* Feature engineering
* Encoding
* Dataset preparation

### 3. Training

`training.py`

Trains the machine learning models and stores the training results.

### 4. Evaluation

`evaluation.py`

Evaluates trained models using classification metrics.

### 5. Prediction

`main.py`

Provides the main prediction workflow using the trained model.

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest
* XGBoost

### Experiment Tracking

* MLflow

### Development

* Jupyter Notebook
* Python Script

---

## 💻 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/danielgracemiracle-ops/uas-credit-score
cd uas-credit-score
```

### 2. Create virtual environment

```bash
python -m venv ai_env
```

Activate the environment:

**Windows**

```bash
ai_env\Scripts\activate
```

**Linux / macOS**

```bash
source ai_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

---

## 📌 Key Results

The final Random Forest model achieved:

* **Accuracy:** 74.96%
* **F1-Score:** 74.94%

The model demonstrated the ability to classify credit score categories using customer financial and credit-related features.

---

## 📚 Learning Outcomes

This project provided practical experience in:

* Data preprocessing
* Feature engineering
* Classification
* Random Forest
* XGBoost
* Model evaluation
* Feature importance analysis
* MLflow experiment tracking
* Modular Python development
* Machine learning pipeline design

The project also provided experience in converting a notebook-based machine learning workflow into a **structured Python pipeline** consisting of separate ingestion, preprocessing, training, evaluation, and prediction modules.

---

## 🚀 Future Improvements

Potential improvements include:

* Hyperparameter optimization
* Cross-validation
* Class imbalance handling
* Explainable AI with SHAP
* REST API deployment
* Interactive prediction dashboard
* Cloud deployment
* Automated model retraining
* Model monitoring

---

## 👨‍💻 Project Role

**Machine Learning / Data Science Developer**

Responsibilities included:

* Data preprocessing
* Feature engineering
* Model development
* Model comparison
* Model evaluation
* Feature importance analysis
* MLflow experiment tracking
* Machine learning pipeline development
