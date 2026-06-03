# Give Me Some Credit: Credit Risk Modeling

This project builds an interpretable credit risk model using the Kaggle **Give Me Some Credit** dataset. The goal is to predict whether a borrower will experience serious delinquency within the next two years.

## Project Overview

The project follows a typical offline credit risk modeling workflow:

- Data cleaning and missing value treatment
- Outlier inspection and percentile capping
- Decision-tree-based binning for continuous variables
- Business-based binning for delinquency count variables
- WOE transformation and IV-based feature selection
- Logistic Regression modeling
- Model evaluation using AUC, KS, and Gini
- Decile/Lift analysis
- Cutoff strategy analysis

## Key Results

| Model | AUC | KS | Gini |
|---|---:|---:|---:|
| Logistic WOE Scorecard | 0.857 | 0.567 | 0.713 |

The highest-risk 10% of borrowers captured 52.82% of all bad borrowers. Rejecting the top 20% highest-risk borrowers captured 71.07% of bad borrowers and reduced the approved population bad rate to 2.41%.

## Files

- `credit_risk_scorecard.ipynb`: Main notebook containing data cleaning, feature engineering, modeling, and evaluation.
- `give_me_some_credit.pdf`: Final project report.
- `cs-training.csv`: Labelled training dataset.
- `cs-test.csv`: Kaggle test dataset without observable target labels.
- `Data Dictionary.xls`: Original variable descriptions.
- `requirements.txt`: Python package requirements.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook credit_risk_scorecard.ipynb
