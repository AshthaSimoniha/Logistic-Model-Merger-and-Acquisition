## **Dataset**

The data has been acquired from funding hence cannot be shared but to understand the code this is the head of the data frame.

![image](https://user-images.githubusercontent.com/85382522/120883633-d8d17180-c5fb-11eb-949b-bcf6b783a765.png)



Brand Name is used as index and MA as the dependent variable. Other columns indicate the independent variables.

## Logistic Model

When the dependent variable is dichotomous, logistic regression is the only regression analysis to use (binary). The logistic regression, like all regression analyses, is a statistical analysis. To characterise data and illustrate the relationship between one dependent binary variable and one or more nominal, ordinal, interval, or ratio-level independent variables, logistic regression is used.

![image](https://user-images.githubusercontent.com/85382522/120883653-03232f00-c5fc-11eb-9dbb-20d397ae3dc0.png)

## Map to Project

- Initially the data is cleared by removing missing values.
- Splitting of the Dataset With scikit learn's (sklearn) train_test_split() into training and test dataset for unbiased evaluation.
- Using sklearn to fit the trainng model on the data set.
- Checking accuracy of the model with accuracy score(sklearn.metrics), confusion matrix and ROC curve.
- Performing logistic regression using statsmodel.
