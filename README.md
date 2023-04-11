# Machine Learning Project

This project involves building and testing machine learning models using the Perceptron Learning Algorithm, Fisher's Linear Discriminant Analysis, and Logistic Regression. The dataset used in this project contains 32 features, and some of the data points have missing feature values that need to be imputed.

**Feature Engineering Tasks:**

Impute missing feature values with the most frequent value for categorical features and the average value for continuous numerical features.
Normalize each feature using (X’ = (X - µ) / σ), where µ represents the mean of feature value and σ represents the standard deviation of feature values.


Part A - **Perceptron Learning Algorithm:**
Learning Task 1: Build a classifier (Perceptron Model - PM1) using the perceptron algorithm and determine whether the dataset is linearly separable. Build another classifier (PM2) by changing the order of the training examples and outline the differences between PM1 and PM2.
Learning Task 2: Build a classifier (PM3) using the perceptron algorithm on normalized data and compare it with PM1.
Learning Task 3: Change the order of features in the dataset randomly and build a classifier (PM4). Determine whether there are any differences in the model and its performance compared to PM1.

Part B - **Fisher's Linear Discriminant Analysis:**
Learning Task 1: Build Fisher's linear discriminant model (FLDM1) on the training data to reduce the 32 dimensional problem to univariate dimensional problem. Find the decision boundary in the univariate dimension using the generative approach assuming Gaussian distribution for both positive and negative classes.
Learning Task 2: Change the order of features in the dataset randomly and build the Fisher's linear discriminant model (FLDM2) on the same training data as in Learning Task 1. Find the decision boundary in the univariate dimension using the generative approach and outline the differences between FLDM1 and FLDM2 and their respective performances.

Part C - **Logistic Regression:**
Learning Task 1: Build a classification model (LR1) using Logistic Regression. Vary the decision probability threshold from 0.5 to 0.3, 0.4, 0.6, and 0.7 and observe the effect on testing accuracy.

Testing:
Use random 67% of the data to train the models and 33% of the data to test the models. For each model, perform at least 10 random training and testing splits of the data and report the results of all these splits along with the average and variance of performance metrics.

Note: The project code and results will be uploaded to GitHub.
