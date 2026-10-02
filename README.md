
# Sentiment_Analysis_for_Customer_Feedback

![image](https://github.com/user-attachments/assets/dd4e8314-2587-49e9-a408-9ad6501d4a48)

## Table of Cotents

- [AliExpress Review Sentiment Classifier Live Demo](#AliExpress-Review-Sentiment-Classifier-Live-Demo)
- [Introduction & Problem Statement](#Introduction-&-Problem-Statement)
- [Project Objectives & Deliverables](#Project-Objectives-&-Deliverables)
- [Dataset Overview](#Dataset-Overview)
- [Tools & Technology Stack](#Tools-&-Technology-Stack)
- [Methodology / Pipeline Overview](#Methodology-/-Pipeline-Overview)
- [Data Loading & Inspection](#Data-Loading-&-Inspection)
- [Data Cleaning](#Data-Cleaning)
- [Label Engineering](#Label-Engineering)
- [Text Preprocessing](#Text-Preprocessing)
- [Train / Test Split](#Train-/-Test-Split)
- [Feature Engineering (Bag of Words & TF-IDF)](#Feature-Engineering (Bag of Words & TF-IDF))
- [Handling Class Imbalance (Train-Only Oversampling)](#Handling-Class-Imbalance (Train-Only Oversampling))
- [Model 1 - VADER (Rule-Based Sentiment)](#Model-1-VADER (Rule-Based Sentiment))
- [Models 2–4 - Naive Bayes, Logistic Regression & Linear SVM](#Models-2–4-Naive-Bayes,-Logistic-Regression-&-Linear-SVM)
- [Model Comparison & Results](#Model-Comparison-&-Results)
- [Hyperparameter Tuning (GridSearchCV Pipeline)](#Hyperparameter-Tuning (GridSearchCV Pipeline))
- [Inference Pipeline](#Inference-Pipeline)
- [Model Deployment](#Model-Deployment)
- [Limitations](#Limitations)
- [Recommendations & Future Work](#Recommendations-&-Future-Work)
- [Conclusion](#Conclusion)



## AliExpress Review Sentiment Classifier Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentimentanalysisforcustomerfeedback-hqdwyycxgcdkc9tmdktgxg.streamlit.app/)

A sentiment analysis tool that classifies AliExpress electronics reviews as positive, negative, or neutral, with support for single reviews and batch (CSV uploads).

## Introduction & Problem Statement
AliExpress is a large global e-commerce platform (a subsidiary of the Alibaba Group) connecting millions of buyers with sellers across categories including electronics, fashion, and home goods. Understanding how customers feel about their purchases beyond the numeric star rating alone is central to catching product-quality issues, informing seller policy, and prioritizing customer-experience improvements.
Manually reading thousands of free-text reviews to gauge sentiment does not scale. This project builds an automated pipeline that classifies each review's text as positive, neutral, or negative, so stakeholders can monitor sentiment trends at scale instead of one review at a time.

## Project Objectives & Deliverables
The objective was to build an end-to-end solution that can extract, preprocess, analyze, and eventually visualize customer sentiment from AliExpress Electronics reviews. This report covers the Data Science workstream: data preparation, feature engineering, model development across multiple algorithms, evaluation, hyperparameter tuning, and deployment packaging.
Key deliverables covered in this report:

●	A cleaned, labeled dataset mapping star ratings to sentiment classes

●	A reusable, negation-preserving text-preprocessing function

●	Two vectorization pipelines: Bag of Words and TF-IDF

●	Four modelling approaches (VADER, Naive Bayes, Logistic Regression, Linear SVM) evaluated across both feature sets — eight combinations in total

●	A leakage-safe hyperparameter-tuning pipeline for the strongest configuration

●	A packaged inference function and serialized model artifact ready for deployment

## Dataset Overview
The raw dataset (data.csv) contains 46,100 AliExpress reviews with the following columns:

<img width="628" height="223" alt="Data Overview" src="https://github.com/user-attachments/assets/e4b581d4-cae4-4b94-b7a7-a5dd65e448ad" />

Star ratings were heavily skewed toward the extremes — the raw distribution was: 5 stars = 22,553; 1 star = 10,433; 4 stars = 1,895; 3 stars = 957; 2 stars = 868. This skew, and how it was addressed, is discussed in Sections 9 and 13.


## Tools & Technology Stack

<img width="627" height="319" alt="Tools   Tech stack" src="https://github.com/user-attachments/assets/1623a9b4-28f5-4b41-8ce2-929a53f20d1b" />

## Methodology / Pipeline Overview
The project followed four broad stages, mirroring the original internship brief:

●	Data Collection — load the raw AliExpress review export and inspect its structure

●	Text Processing — clean the data, derive sentiment labels from star ratings, and normalize review text

●	Modelling — build a rule-based baseline (VADER) and three supervised classifiers (Naive Bayes, Logistic Regression, Linear SVM), each on two feature representations

●	Model Evaluation, Tuning & Deployment — compare all approaches on a common held-out test set, tune the strongest configuration, then package it for inference

The train/test split happens before any vectorization or class balancing, and oversampling is applied only to the training data after vectorizing it. This avoids leaking duplicated rows into the test set — the test set is left in its natural, imbalanced state throughout, so the accuracy and F1  reflect real-world performance rather than an inflated, balanced-test-set estimate.

## Data Loading & Inspection
The raw CSV was loaded into a pandas DataFrame and the columns were renamed to short, consistent, lowercase names for easier downstream handling.

<img width="420" height="143" alt="Data loading  inspection 1" src="https://github.com/user-attachments/assets/b41c522c-578c-453b-9fc0-f93a775821c5" />

An initial df.info() call confirmed the dataset shape (46,100 rows, 7 columns) and revealed that the text column had substantial missing data — only 36,717 of 46,100 rows had review text populated.

<img width="293" height="241" alt="Data loading  inspection 2" src="https://github.com/user-attachments/assets/ccbd2039-65d6-4315-aaa5-da88bbfda62e" />










