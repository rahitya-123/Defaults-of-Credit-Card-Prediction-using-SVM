# Credit Card Default Prediction using Support Vector Machine (SVM)

A machine learning project that predicts credit card payment defaults using Support Vector Machine classification with comprehensive data preprocessing and model optimization.

## 📊 Project Overview

This project analyzes credit card client data to predict whether a customer will default on their next payment. The model uses various customer attributes including demographics, payment history, and billing information to make accurate predictions.

## 🎯 Objective

- Predict credit card payment defaults using historical customer data
- Implement and optimize Support Vector Machine for binary classification
- Handle real-world data challenges including missing values and categorical encoding
- Achieve balanced performance on both default and non-default predictions

## 📋 Dataset Information

**Source**: Default of Credit Card Clients Dataset
- **Total Records**: 30,000 customers
- **Features**: 24 attributes (after preprocessing)
- **Target Variable**: Default payment next month (Binary: 0=No Default, 1=Default)

### Key Features:
- **LIMIT_BAL**: Credit limit amount
- **Demographics**: Sex, Education, Marriage, Age
- **Payment History**: PAY_0 to PAY_6 (6 months of payment status)
- **Bill Amounts**: BILL_AMT1 to BILL_AMT6 (6 months of billing)
- **Payment Amounts**: PAY_AMT1 to PAY_AMT6 (6 months of payments)

### Feature Encoding:
- **SEX**: 1=Male, 2=Female
- **EDUCATION**: 1=Graduate, 2=University, 3=High School, 4=Others
- **MARRIAGE**: 1=Married, 2=Single, 3=Others
- **Payment Status**: -1=Pay Duly, 1-9=Payment Delay (months)

## 🛠️ Technical Implementation

### Libraries Used
```python
pandas              # Data manipulation and analysis
numpy               # Numerical computing
matplotlib          # Data visualization
scikit-learn        # Machine learning algorithms
 - SVM             # Support Vector Machine
 - GridSearchCV    # Hyperparameter tuning
 - train_test_split# Data splitting
 - preprocessing   # Data scaling and encoding
 - metrics         # Model evaluation
