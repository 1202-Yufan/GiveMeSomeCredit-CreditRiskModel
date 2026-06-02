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

The final Logistic WOE model achieved the following validation performance:

| Model | AUC | KS | Gini |
|---|---:|---:|---:|
| Logistic WOE Scorecard | 0.857 | 0.567 | 0.713 |

The Decile/Lift analysis shows that the highest-risk 10% of borrowers captured 52.82% of all bad borrowers. In the cutoff strategy analysis, rejecting the top 20% highest-risk borrowers captured 71.07% of bad borrowers and reduced the approved population bad rate to 2.41%.

## Files

- `programe.ipynb`: Main notebook containing data cleaning, feature engineering, modeling, and evaluation.
- `give_me_some_credit.pdf`: Final project report.
- `cs-training.csv`: Training dataset.
- `cs-test.csv`: Kaggle test dataset.
- `Data Dictionary.xls`: Original variable descriptions.
- `requirements.txt`: Python package requirements.

## Notes

The official Kaggle test set does not contain observable target labels, so model validation is performed using a stratified hold-out validation set split from the training data.

This project is intended as an academic and portfolio project for credit risk modeling practice.
