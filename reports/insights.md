# Churn Insights Report

## Model Performance (Test Set)
### Logistic Regression
- Accuracy: 0.80
- Precision: 0.65
- Recall: 0.56
- F1 Score: 0.60
- ROC-AUC: 0.84

### Random Forest
- Accuracy: 0.80
- Precision: 0.69
- Recall: 0.49
- F1 Score: 0.57
- ROC-AUC: 0.84

## Top 5 Random Forest Features
0.099532
0.097318
0.083398
0.028119
0.026232

## Top 5 Logistic Regression Coefficients
                    feature  coefficient
          Contract_Two year    -1.315145
                     tenure    -1.214847
InternetService_Fiber optic     1.058648
      customerID_0607-DAAHE     0.988151
      customerID_6323-AYBRX     0.988040

## Segmented Churn Rates
- Contract: Higher churn in month-to-month contracts.
- Senior Citizen: Higher churn among senior citizens.
- Tenure: Customers with shorter tenure churn more.

See plots in `app/static/plots/`.
