# Supervised Machine Learning Homework - Predicting Credit Risk

In this assignment, I will be building a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

I will be using this data to create machine learning models to classify the risk level of given loans. I will be comparing the Logistic Regression and Random Forest Classifier model.

## Instructions

### Retrieve the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

I will be using an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

Create a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, create a testing set from the 2020 loans, also using `pd.get_dummies()`.

## Consider the models

I will be creating and comparing two models on this data: a logistic regression, and a random forests classifier. Before I create, fit, and score the models, I made a prediction as to which model will perform better. 

## Fit a LogisticRegression model and RandomForestClassifier model

Create a LogisticRegression model, fit it to the data, and print the model's score. I did the same for a RandomForestClassifier. 

## Predictions 

The Random Forest Classifier model will perform better on the unscaled data based on the fact that it has higher train and test accuracies. Also due in part to it's model complexity. 

The Logistic Regression model will perform better on the scaled data due to the sensitivity to the range of the data points.

## Comparisons of Models - Both Unscaled and Scaled Data

Below is a comparison between the models on both unscaled and scaled data along with the accuracy data.

![Screen Shot 2021-08-21 at 9 34 11 AM](https://user-images.githubusercontent.com/78628287/130323465-1182a61f-e307-4d44-8d6d-15cb9ddad177.png)

## Unscaled Logistic Regression:

* Training Data Score: 0.6677339901477832
* Testing Data Score: 0.6133560187154402

![Screen Shot 2021-08-21 at 9 52 38 AM](https://user-images.githubusercontent.com/78628287/130323924-abce1baf-c1d3-454f-af5b-dafa8f783bc5.png)

## Scaled Logistic Regression:

* Training Data Score: 0.6858784893267652
* Testing Data Score: 0.6816248404934071

![Screen Shot 2021-08-21 at 9 53 28 AM](https://user-images.githubusercontent.com/78628287/130323957-14ee6c2e-e07f-4075-ab07-a5b59632218f.png)

## Unscaled Random Forest Classifer:

* Training Score: 1.0
* Testing Score: 0.7103360272224585

![Screen Shot 2021-08-21 at 9 54 52 AM](https://user-images.githubusercontent.com/78628287/130323974-8a881db0-5c58-4729-aba9-3a0c28e8bd86.png)

## Scaled Random Forest Classifer:

* Training Score: 1.0
* Testing Score: 0.7096980008507018

![Screen Shot 2021-08-21 at 9 55 22 AM](https://user-images.githubusercontent.com/78628287/130323984-87fd18ae-22e8-4ccf-82cd-8aaacdf95301.png)
