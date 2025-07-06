# 🏠 Home Loan Prediction

This project uses machine learning techniques to predict whether a loan application should be approved or rejected based on the applicant’s financial and personal information. It aims to automate loan approval processes for financial institutions, improving speed and accuracy.

## 🧠 Problem Statement

Banks and financial institutions receive thousands of loan applications daily. Evaluating these manually is time-consuming and prone to human error. Using predictive models, we can determine the likelihood of loan approval based on applicant data such as income, credit history, employment status, etc.

## ✅ Features

- Clean and preprocess real-world loan applicant data
- Train multiple classification algorithms
- Evaluate model performance using accuracy, precision, recall, and F1-score
- Visualize important patterns and model insights
- Predict whether a loan should be approved (`Y`) or not (`N`)

## 📦 Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

## 📊 Dataset

The dataset includes the following key features:

- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- `Loan_Status` (Target)

> Dataset Source: [Loan Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

## 🧪 Models Used

We tested several machine learning models, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The best model was selected based on overall accuracy and generalization on test data.

## 📈 Results

- Best Accuracy: ~80%
- Confusion matrix and classification report used for evaluation
- Feature importance visualized to understand key drivers of loan approval


