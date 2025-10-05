# Online Shopping Intention Analysis

# Project Overview

This project predicts whether a user visiting an e-commerce website will make a purchase. It uses historical user behavior data to perform:

- Exploratory Data Analysis (EDA)
- Data preprocessing (encoding, scaling)
- Machine learning model training
- Performance evaluation

The goal is to understand user behavior and predict purchase intent.

# Dataset

**Source:** Kaggle — Online Shoppers Intention Dataset

**Target Variable:** `Revenue` (0 = No purchase, 1 = Purchase)

**Key Features:**

| Feature | Description |
|---------|-------------|
| ProductRelated_Duration | Total time spent on product pages |
| BounceRates | Fraction of visitors leaving after one page |
| ExitRates | Fraction of exits per page |
| PageValues | Value of pages contributing to revenue |
| VisitorType | New or Returning visitor |
| Weekend | Whether session occurred on weekend |
| Month | Month of the visit |

# Project Steps

## 1. Data Loading & Inspection

- Load dataset into a Jupyter Notebook
- Inspect columns, data types, and missing values

## 2. Exploratory Data Analysis (EDA)

- Target distribution plots (buyers vs non-buyers)
- Numeric and categorical feature analysis
- Correlation heatmap

## 3. Data Preprocessing

- Handle missing values
- Encode categorical features
- Scale numeric features
- Train/Test split

## 4. Model Training

- Train Random Forest Classifier
- Evaluate using accuracy, confusion matrix, and classification report

## 5. Prediction

- Predict purchase intent for new user sessions

# Usage

Clone the repository and run the notebook:

