
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

## Data Cleaning
A closer look at missing values confirmed the text column was the primary source of gaps (9,383 missing values), with a handful of missing username and location entries as well.

<img width="188" height="168" alt="daa cleaning 1" src="https://github.com/user-attachments/assets/dbdaaf54-006e-417e-8a56-b8b051a5b32f" />


Since review text is the only usable model input, rows with any missing values were dropped entirely rather than imputed. This reduced the working dataset from 46,100 to 36,706 rows, all fully populated.

<img width="166" height="161" alt="daa cleaning 2" src="https://github.com/user-attachments/assets/131f6123-5327-450d-88d6-7c9709835e67" />

## Label Engineering (Rating → Sentiment)
The dataset has no direct sentiment label — only a 1–5 star rating. Ratings were mapped to three sentiment classes: 1–2 stars → negative, 3 stars → neutral, 4–5 stars → positive.

<img width="173" height="137" alt="label engineering 1" src="https://github.com/user-attachments/assets/9d40c6fb-4e05-4465-a1d1-fff8b66f0376" />

This mapping surfaces the central data-quality challenge of the project: the resulting classes are highly imbalanced — 24,448 positive vs. 11,301 negative vs. only 957 neutral reviews (2.6% of the data). This imbalance persists into the test set and is the main driver of every model's weak neutral-class performance.

<img width="352" height="149" alt="label engineering 2" src="https://github.com/user-attachments/assets/cd9a8432-c7ba-4472-9f6a-3f7dd07c7f5b" />

## Text Preprocessing
Before vectorizing review text, a stopword-removal function strips out uninformative words (e.g. “the”, “is”, “and”) using NLTK's tokenizer and English stopword list.

<img width="402" height="91" alt="Text processing" src="https://github.com/user-attachments/assets/dbdcf74e-aeda-4f11-bb95-6f2caa78b839" />

This function was applied across all 36,706 reviews to create a new text_without_stopwords column, used as the input to the Bag-of-Words/TF-IDF pipelines and to one of the two VADER variants.

## Train / Test Split
The cleaned dataset was split into training and test sets using an 80/20 split (test_size=0.2, random_state=42) before any vectorization or balancing took place. Splitting first, and applying oversampling only afterward on the training side, is what keeps the test set an honest, unmodified sample of real review traffic.

<img width="311" height="86" alt="train test split" src="https://github.com/user-attachments/assets/43d78556-9c87-4e49-8fdf-561513f8dd0e" />

## Feature Engineering (Bag of Words & TF-IDF)
Machine learning models require numerical input, so the cleaned review text was converted into two different numerical representations for comparison:

●	Bag of Words (BoW) — counts how often each word (and word pair, via ngram_range=(1,2)) appears in a review, ignoring word order.

●	TF-IDF (Term Frequency – Inverse Document Frequency) — down-weights words that appear across most reviews (e.g. “product”) and up-weights words distinctive to a given review.

Both vectorizers were fit on the training text only, then used to transform the test text, to avoid data leakage. Unigrams and bigrams together (ngram_range=(1,2)) let the models pick up two-word phrases like “not good” or “fast delivery”, not just isolated words.

<img width="476" height="191" alt="feature engineering" src="https://github.com/user-attachments/assets/de0eeb37-c607-47da-8f08-1e844ed5d1a0" />

## Handling Class Imbalance (Train-Only Oversampling)
Because the neutral class made up only a small fraction of reviews, and this fraction shrinks further once the data is split (784 of 29,364 training rows, or 2.7%), training directly on the raw distribution would bias any classifier toward never predicting neutral. Random Oversampling was applied to the vectorized training features only, duplicating minority-class examples until all three classes were equally represented in the training set — the test set was left untouched.

<img width="468" height="145" alt="Handling class inmalance" src="https://github.com/user-attachments/assets/aa9b8eaa-2cf5-4bd3-89ff-d9b5a3dda91a" />


<img width="563" height="256" alt="Handling class inmalance 2" src="https://github.com/user-attachments/assets/b93e9736-3cc7-497b-b2a4-86c3837439d2" />

## Step 8 — Model 1: VADER (Rule-Based Sentiment)
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a pretrained, lexicon-based sentiment model — it requires no training data and scores each word in a sentence against a dictionary of sentiment intensities, combining those into an overall “compound” score. This iteration uses a three-way threshold intended to add a neutral band around a compound score of 0.05:

<img width="429" height="187" alt="model 1 vader" src="https://github.com/user-attachments/assets/e04bb3bb-cb12-41ec-b89b-2e0f0c15caa1" />

Note for the technical reviewer: as written, the elif compound_score <= 0.05 branch fires on everything the first condition didn't catch (i.e. every score below 0.05 already satisfies “≤ 0.05”), so the else: neutral branch can never execute. This is why VADER's neutral-class precision and recall are exactly 0.00 in both classification reports below — not a lexicon weakness, but the threshold logic never actually reaching the intended neutral case. Fixing it would require a genuine band, e.g. treating scores between -0.05 and 0.05 as neutral, before the negative check.
VADER was applied to two versions of the test set — raw text and stopword-stripped text — and scored well above the previous iteration's ~55% because this test set is far more skewed toward positive/negative reviews (only 2.4% neutral) than a hypothetical balanced one, which flatters any model that defaults to those two classes.

## Models 2–4: Naive Bayes, Logistic Regression & Linear SVM

Three supervised classifiers were trained on the balanced training features (Section 13) — one Bag-of-Words model and one TF-IDF model each — and evaluated against the untouched, naturally-imbalanced test labels.

Multinomial Naive Bayes:
mnb_bow = MultinomialNB(); mnb_tfidf = MultinomialNB()

mnb_bow.fit(X_train_bow_res, y_train_bow_res)

mnb_tfidf.fit(X_train_tfidf_res, y_train_tfidf_res)

Logistic Regression:

logreg_bow = LogisticRegression(); logreg_tfidf = LogisticRegression()

logreg_bow.fit(X_train_bow_res, y_train_bow_res)

logreg_tfidf.fit(X_train_tfidf_res, y_train_tfidf_res)

Linear SVM:

svm_bow = LinearSVC(); svm_tfidf = LinearSVC()

svm_bow.fit(X_train_bow_res, y_train_bow_res)

svm_tfidf.fit(X_train_tfidf_res, y_train_tfidf_res)

Predictions from all three classifiers, on both feature sets, were attached back onto the test DataFrame alongside the two VADER outputs and the true label, giving a single table to evaluate all eight approaches against a common ground truth.

## Model Comparison & Results
All eight pipelines were scored on the identical held-out test set of 7,342 reviews (2,301 negative, 173 neutral, 4,868 positive) using accuracy and per-class precision/recall/F1.

<img width="640" height="296" alt="model comparison 1" src="https://github.com/user-attachments/assets/308d6415-4fd4-45f2-8538-5de5deb06583" />

VADER — raw text vs. stopword-removed text

<img width="384" height="207" alt="vader - raw text vs stopwords" src="https://github.com/user-attachments/assets/f81ffb3f-03af-47d6-8c6b-41ba6505ac94" />

<img width="432" height="201" alt="vader - raw text vs stopwords 2" src="https://github.com/user-attachments/assets/255a4cf6-a4f5-4d13-af28-7660a0eb6ab3" />























































