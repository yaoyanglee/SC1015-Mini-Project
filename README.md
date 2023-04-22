# Price prediction of used devices

School of Computer Science and Engineering \
Nanyang Technological University \
Lab: B124 \
Team: 5

Members:

1. Lee Yao Yang
2. Tang Ken Yi
3. Ang Zi Yang, Alvis

---

### Introduction

This mini project aims to predict the price of used devices (Phones and Tablets) of various brands based on features such as Brand, Weight, Screen Size, etc.

This repository contains the code and dataset used throughout the project.
The notebook contains explanations of the code snippets, whereas this readme will contain explanations on the general ideas and motivations behind the steps we have taken.
The dataset used for this project can be found [here](https://www.kaggle.com/code/kavya2099/used-phone-price-prediction/notebook).

---

### Table of Contents

1. [Problem Formulation](#1-Problem-Formlation)
2. [Data Preparation and Cleaning](#2-Data-Preparation-and-Cleaning)
3. [Exploratory Data Analysis](#3-Exploratory-Data-Analysis)
4. [Feature selection](#4-Feature-Selection)
5. [Models](#5-Models)
6. [Data Driven Insights and Conclusion](#6-Data-Driven-Insights-and-Conclusion)

---

### 1. Problem Formulation

In today's technologically enabled world, smartphones and tablets have become items that we rely on. It has evolved from a luxury to a necessity that allows not only for communication but also for learning and payments among other day-to-day activities. Suffice to say it has become irreplaceable. But as their use becomes more ubiquitous e-waste recycling has become a necessity. Due to hazardous materials in e-waste, a lack of recycling can cause environmental contamination. At the same time, as the usage of phones and tablets increases, the use of precious materials increases, leading to a fall in such resources. Thus recycling has become more important than ever.

In 2016, the market for used devices was valued at $17 billion dollars, with a 50% year-on-year growth in units. The market for used devices is one that is growing and one that we believe will impact many in the coming years. Thus we sought to help buyers and sellers estimate the value of the price of the used devices they are intending to buy or sell.

The question we have thus arrived at is "What is the likely price of my phone?"

This question would allow buyers and sellers to gain an understanding of the market values based on the specifications of the phones, allowing them to make the best decision when buying their phones.

### 2. Data Preparation and Cleaning

For data preparation and cleaning, we removed the NULL values and also one-hot encoded the categorical values

1. Removal of NULL values \
   When we encountered NULL values, there were 2 options that we could take. First was to replace the NULL values with the mean or median. Second was to remove the NULL values.

   For the first case, replacing the NULL values with the mean or median was not a good option mainly due to the skew of some of the variables. During our EDA process, we found that some variables such as weight were skewed heavily. The median and mean would differ in this case. It would be difficult to determine which would more accurately represent the data. This is because there could be design or economic considerations in the development of these smart devices which result in a skew in the data. It would be difficult to determine if mean or median is best used to represent the entire dataset.

   Furthermore, when looking at the NULL values, they represent at maximum only 5% of the data. This is a relatively small percentage given our large data size.

   Due to the small number of outliers and the difficulty in ascertaining whether mean or median is more representative of the data, our group has decided to remove the outliers.

2. One hot encoding \
   Categorical variables were present in the dataset as well. Some examples were whether the phone has 4G or 5G connectivity. Encoding them numerically does represent the categorical data in a numerical form, but when performing machine learning these values would be treated as numerical continuous variables. This would result in accurate variables being used as predictors for the model. Thus we performed one hot encoding on the data using the pandas method get_dummies().

### 3. Exploratory Data Analysis

The main aim of this section was to understand the distribution of each variable and find relationships between the independent and dependent variables.
We used this section to determine 4 things.

1. Removal of NULL values \
   The explanation has been given above. In short, we found that the data was skewed, and it is difficult to determine whether the mean or median best represented the variable for the entire dataset. Thus we decided to remove NULL values.

2. Usage of Min Max Scaler vs Standard Scaler \
   Standard Scaler is used when the data is normal, and Min Max Scaler when we know the minimum and maximum value of the variables from domain knowledge. Looking at the histograms, we find that not all variables follow a normal distribution, thus a Min Max Scaler was used.

   Furthermore, the minimum and maximum values of a variable in the used devices are known, as they are recorded within the dataset, as well as in research literature or product information pages. Thus a Min Max Scaler was used.

3. Importance of each variable \
   From our EDA, we found that a majority of data points for RAM were clustered together at 4GB of RAM. However, upon further analysis in our bivariate data analysis, we discovered that there was indeed a relationship between the variable and the price of used devices. Thus we decided to keep the variable. A possible explanation for this would be manufacturing considerations such as cost, and knowledge from market research among many other factors. We also decided to keep this data, as it was representative of the data. This also reinforced the idea of removing the NULL values, as there were too many factors that were unknown to us which could have contributed to the distribution of the data with respect to each variable. We would not be able to determine if mean or median best represented the dataset for a particular variable, hence justifying the removal of the NULL values.

4. Confirmation of a relationship between the variables \
   This reassured us that the variables are related to the dependent variable, used price, which we are planning to predict. Allowing us to proceed with our regression analysis.

### 4. Feature selection

To determine which are the most important variables, we need to first identify which selection model we should use. When selecting the model, we need to look at the type of data, number of features, and complexity of relationships. Looking at the pairplots, we identify that the relationship is non-linear. Since the relationship between the features and the target variable (normalized_used_price) is non-linear and complex, we will use Mutual Information Regression (MIR) to find the rank of the variables. We then use kbest to take the top 5 variables from the results of the MIR. Here, we have identified our top 5 variables.

Since our dataset consisted of many variables, it had a high dimensionality, which would lead to overfitting, increased computational complexity, and feature redundancy. To address this, we used PCA (Principal Component Analysis), which reduces the number of features while preserving the important information. Here, we use PCA to reduce the dimensionality of the dataset, improving the accuracy and efficiency of the model.

### 5. Models

1. Decision Tree Regression \
   There were 2 reasons which motivated us to use this model.

   First, a decision tree is not largely influenced by outliers or missing values and can handle both categorical and continuous variables. From our EDA, we found that not only were there missing values that we removed, there were also outliers that we kept. The outliers were kept as we believed that the outliers were natural variations when it came to the production of devices, and that the removal of them would significantly alter the representation of the dataset. Thus although we kept the outliers due to it being representative of the population of used devices, we did not want them to heavily affect the accuracy of the model in predicting used price.

   Second, a decision tree can handle both categorical and continuous variables. Our dataset consists of a mix of categorical and continuous variables, thus we decided that it was a good model that could give accurate predictions.

2. Random Forest Regression \
   Random Forest is a type of Ensemble learning, which involves combining predictions from 2 or more models. Random Forest uses a specific case of Ensemble learning, known as Bootstrap aggregation or Bagging. In Bagging, the model is trained using different samples of the same training dataset. The predictions made by the ensemble of models are then combined using simple statistics such as voting or averaging.

   This model was chosen because the regression models we have learnt thus far in the course have been making predictions based only on a randomised selection of training data from the original data. However, such a randomised selection could result in the data being biased. Furthermore, it also mitigates against overfitting of the model and a reduction in variance. There were a few variables with a particularly high standard deviation in our dataset, thus we sought to use a model which would reduce it, allowing for a more accurate prediction.

3. Linear Regression \
   Linear Regression is a simple but effective way to analyse the used price with respect to multiple variables. After using the regression models above, we were curious to see if the used price had a linear relationship with the dependent variables. Finding this is important, as a simple linear relationship to the predictors would allow buyers and sellers to easily understand the price of a used device based on comparisons to other devices. This allows for an intuitive understanding of the price. Furthermore, Linear Regressions are less computationally expensive compared to Decision Trees or Ensemble learning models. If a simple and accurate linear relationship could be established, the prediction would remain accurate and much faster with a larger dataset in comparison to the one we are working with.

   Due to high R squared value and low MSE, we decided to further our analysis of the linear regression model using Ridge and LASSO regressions.

4. Ridge Regression \
   Ridge Regression is a model tuning method used to analyse any data that suffers from multicollinearity. Multicollinearity is the case when 1 predictor is highly linearly related to another predictor. If simple linear regression is to study the effects of how each predictor affects the response, used price, then multicollinearity undermines this principle, as the predictors are not independent from each other. The regression coefficients are thus not uniquely determined, as changes in one feature results in changes in the response variable and another predictor variable. This results in an inaccurate regression model.

   Thus we aim to use Ridge Regression which performs L2 regularization to reduce multicollinearity by taking the square of the coefficients as the penalty factor, as the effects of insignificant predictors and multicollinear predictors are reduced.

5. Lasso Regression \
   Lasso Regression is similar to Ridge Regression with the difference that instead of taking the square of the coefficients as the penalty factor, the magnitude of the coefficients are used instead, which is also known as L1 regularization. The reasons for using Lasso Regression are similar to Ridge Regression. We chose this model simply to examine the difference between the L1 and L2 regularization on the accuracy of the model.

Both Lasso and Ridge Regression are forms of hyperparameter tuning in order to derive a better linear regression model.

6. Support Vector Regression (SVR) \
   SVR tries to find a line, in this case called a Hyperplane, to fit the data. Its aim is to find a hyperplane that distinctly classifies the data points on either side of the hyperplane. We decided to include this regression model, as we saw some non-linear relationships within our data during EDA. Using strictly a linear model results in us having a less than accurate model. We therefore decided to fit a SVR on our dataset.

### 6. Data Driven Insights and Conclusions

After implementing and evaluating the 6 models, it was interesting to see that despite the lack of observable linear relationships during the EDA, after performing one hot encoding, normalisation and PCA, the data actually fits best to a linear regression. Specifically, to the Ridge Regression model. This suggests that after hyperparameter tuning to remove multicollinearity of the data, the predictors actually varied linearly to the used device price.

A corollary to this was the importance of a good data preparation pipeline. In this mini project it is observable that the data did not all follow a linear relationship. But through appropriate data preparation, we were able to normalize and select only important information. This resulted in accurate models which were unexpected given our initial observations in EDA. We have thus arrived at a good model and data preparation pipeline for accurately predicting used device prices, despite the non-linear relationships.

We learned that if the relationship between variables are non-linear, we can use mutual info regression to estimate the mutual information between two variables, which can capture non-linear dependencies. We then selected the best variables based on the MIR scores. After that, we learned that we could use PCA to reduce dimensionality and computational complexity.

<<<<<<< Updated upstream
In conclusion, we explored different ML models in order to predict the price of used devices based on various features. Our data-driven insights can help buyers and sellers make informed decisions in the competitive used device market, ensuring that they get the best value for their money. In fact, the models follow a simple linear regression the best. Though this does not naturally follow from the raw unprocessed data, an intuition can be made, allowing for a simple understanding of used price based on current phone specifications. The model can also be easily updated with new data as the market evolves, making it a valuable tool for all stakeholders in the used device ecosystem. Future work could involve exploring additional features, such as battery life, to further improve the model's accuracy and integrating the model into a user-friendly application for easy access by consumers. We can also attempt to evaluate whether or not a used phone with a particular price is "worth it" or not. With the data, we predicted that we could develop a simple platform for users to evaluate if their price for a particular phone is reasonable or not, and thus "worth it" or not.
=======
In conclusion, we explored different ML models in order to predict the price of used devices based on various features. Our data-driven insights can help buyers and sellers make informed decisions in the competitive used device market, ensuring that they get the best value for their money. In fact, the models follow a simple linear regression the best. Though this does not naturally follow from the raw unprocessed data, an intuition can be made, allowing for a simple understanding of used price based on current phone specifications. The model can also be easily updated with new data as the market evolves, making it a valuable tool for all stakeholders in the used device ecosystem. Future work could involve exploring additional features, such as battery life, to further improve the model's accuracy and integrating the model into a user-friendly application for easy access by consumers. We can also attempt to evaluate whether or not a used phone with a particular price is "worth it" or not. With the data we predicted we could develop a simple platform for users to evaluate if their price for a particular phone is reasonable or not, and thus "worth it" or not.

### 7. References

1. https://www.upgrad.com/blog/pros-and-cons-of-decision-tree-regression-in-machine-learning/
2. https://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/
3. https://towardsdatascience.com/random-forest-regression-5f605132d19d
4. https://www.mygreatlearning.com/blog/what-is-ridge-regression/
5. https://medium.com/future-vision/collinearity-what-it-means-why-its-bad-and-how-does-it-affect-other-models-94e1db984168
6. https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression
>>>>>>> Stashed changes
