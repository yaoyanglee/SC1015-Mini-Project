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

In today's technologically enabled world, smartphones and tablets have become items that we rely on. It has evolved from a luxury to a necessity that allows not only for communication, but also for learning, payments among other day to day activities. Suffice to say it has become irreplacable. But as their use becomes more ubiquitous e-waste recycling has become a necessity. Due to hazardous materials in e-waste, lack of recycling can cause environmental contamination. At the same time, as usage of phones and tablets increase, the use of precious materials increases, leading to a fall in such resources. Thus recycling has become more important than ever.

In 2016, the market for used devices was valued at $17 billion dollars, with a 50% year on year growth in units. The market for used devices is one that is growing and one that we believe will impact many in the coming years. Thus we sought to help buyers and sellers estimate the value of the price of the used devices they are intending to buy or sell.

The question we have thus arrived at is "What is the likely price of my phone?"

This question would allow buyers and sellers to gain an understanding of the market values based on the specifications of the phones, allowing them to make the best decision when buying their phones.

### 2. Data Preparation and Cleaning

For data preparation and cleaning, we removed the NULL values and also one got encoded the categorical values

1. Removal of NULL values
   When we encountered NULL values, there were 2 options that we could take. First was to replace the NULL values with the mean or median. Second was to remove the NULL values.

   For the first case, replacing the NULL values with the mean or median was not a good option mainly due to the skew of some of the variables. During our EDA process, we found that some variables such as weight is skewed heavily. The median and mean would differ in this case. It would be difficult to determine which would more accurately represent the data. This is because there could be design or economic considerations in the development of these smart devices which result in a skew in the data. It would be difficult to determine if mean or median is best used to represent the entire dataset.

   Furthermore, when looking at the NULL values, they makeup at maximum only 5% of the data. This is a relatively small percentage given our large data size.

   Due to the small number outliers and the difficulty in ascertaining whether mean or median is more representative of the data, our group has decided to remove the outliers.

2. One hot encoding
   Categorical variables were present in the dataset as well. Some exaples were whether the phone has 4G or 5G connectivity. Encoding them numerically does represent the categorical data in a numerical form, but when performing machine learning these values would be treated as numerical continuous variables. This would result in accurate variables being used as predictors for the model. Thus we performed one hot encoding on the data using the pandas method get_dummies().

### 3. Exploratory Data Analysis

### 4. Feature selection

### 5. Models

### 6. Data Driven Insights and Conclusions
