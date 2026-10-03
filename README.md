# E-COMMERCE PRODUCT REVIEW SENTIMENT ANALYZER

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
- [Hyperparameter Tuning (GridSearchCV Pipeline)](#Hyperparameter-Tuning-(GridSearchCV-Pipeline))
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
The objective was to build an end-to-end solution that can extract, preprocess, analyze customer sentiment from AliExpress Electronics reviews. This report covers the Data Science workstream: data preparation, feature engineering, model development across multiple algorithms, evaluation, hyperparameter tuning, and deployment packaging.
Key deliverables covered in this report:

●	A cleaned, labeled dataset mapping star ratings to sentiment classes

●	Two vectorization pipelines: Bag of Words and TF-IDF

●	Four modelling approaches (VADER, Naive Bayes, Logistic Regression, Linear SVM) evaluated across both feature sets — eight combinations in total

●	Hyperparameter-tuning pipeline for the strongest configuration

●	A packaged inference function and serialized model artifact ready for deployment

## Dataset Overview
The raw dataset (data.csv) contains 46,100 AliExpress reviews with the following columns:

<img width="628" height="223" alt="Data Overview" src="https://github.com/user-attachments/assets/e4b581d4-cae4-4b94-b7a7-a5dd65e448ad" />

Star ratings were heavily skewed toward the extremes — the raw distribution was: 5 stars = 22,553; 1 star = 10,433; 4 stars = 1,895; 3 stars = 957; 2 stars = 868.

## Tools & Technology Stack

<img width="627" height="319" alt="Tools   Tech stack" src="https://github.com/user-attachments/assets/1623a9b4-28f5-4b41-8ce2-929a53f20d1b" />

## Methodology / Pipeline Overview
The project followed four broad stages, mirroring the original internship brief:

●	Data Collection — load the raw AliExpress review export and inspect its structure

●	Text Processing — clean the data, derive sentiment labels from star ratings, and normalize review text

●	Modelling — build a rule-based baseline (VADER) and three supervised classifiers (Naive Bayes, Logistic Regression, Linear SVM), each on two feature representations

●	Model Evaluation, Tuning & Deployment — compare all approaches on a common held-out test set, tune the strongest configuration, then package it for inference

A key methodology to note, the train/test split happens before any vectorization or class balancing, and oversampling is applied only to the training data after vectorizing it. This avoids leaking duplicated rows into the test set — the test set is left in its natural, imbalanced state throughout, so the accuracy and F1  reflect real-world performance rather than an inflated, balanced-test-set estimate.

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
The dataset has no direct sentiment label — only a 1–5 star rating. Ratings were mapped to three sentiment classes: 1–2 stars: "negative", 3 stars: "neutral", 4–5 stars: "positive".

<img width="173" height="137" alt="label engineering 1" src="https://github.com/user-attachments/assets/9d40c6fb-4e05-4465-a1d1-fff8b66f0376" />

This mapping surfaces the central data-quality challenge of the project, the resulting classes are highly imbalanced — 24,448 positive vs. 11,301 negative vs. only 957 neutral reviews (2.6% of the data). This imbalance persists into the test set.

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

●	Bag of Words (BoW) - counts how often each word (and word pair, via ngram_range=(1,2)) appears in a review, ignoring word order.

●	TF-IDF (Term Frequency – Inverse Document Frequency) — down-weights words that appear across most reviews (e.g. “product”) and up-weights words distinctive to a given review.

Both vectorizers were fit on the training text only, then used to transform the test text, to avoid data leakage. Unigrams and bigrams together (ngram_range=(1,2)) let the models pick up two-word phrases like “not good” or “fast delivery”, not just isolated words.

<img width="476" height="191" alt="feature engineering" src="https://github.com/user-attachments/assets/de0eeb37-c607-47da-8f08-1e844ed5d1a0" />

## Handling Class Imbalance (Train-Only Oversampling)
Because the neutral class made up only a small fraction of reviews, and this fraction shrinks further once the data is split (784 of 29,364 training rows, or 2.7%), training directly on the raw distribution would bias any classifier toward never predicting neutral. Random Oversampling was applied to the vectorized training features only, duplicating minority-class examples until all three classes were equally represented in the training set  the test set was left untouched.

<img width="468" height="145" alt="Handling class inmalance" src="https://github.com/user-attachments/assets/aa9b8eaa-2cf5-4bd3-89ff-d9b5a3dda91a" />


<img width="563" height="256" alt="Handling class inmalance 2" src="https://github.com/user-attachments/assets/b93e9736-3cc7-497b-b2a4-86c3837439d2" />

## Step 8 — Model 1: VADER (Rule-Based Sentiment)
VADER (Valence Aware Dictionary and Sentiment Reasoner) is a pretrained, lexicon based sentiment model, it requires no training data and scores each word in a sentence against a dictionary of sentiment intensities, combining those into an overall “compound” score. This iteration uses a three-way threshold intended to add a neutral band around a compound score of 0.05:

<img width="429" height="187" alt="model 1 vader" src="https://github.com/user-attachments/assets/e04bb3bb-cb12-41ec-b89b-2e0f0c15caa1" />

VADER was applied to two versions of the test set raw text and stopword-stripped text and scored well.

## Models 2–4: Naive Bayes, Logistic Regression & Linear SVM

Three supervised classifiers were trained on the balanced training features, one Bag-of-Words model and one TF-IDF model each, and evaluated against the untouched, naturally-imbalanced test labels.

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

a. VADER on raw text

<img width="384" height="207" alt="vader - raw text vs stopwords" src="https://github.com/user-attachments/assets/f81ffb3f-03af-47d6-8c6b-41ba6505ac94" />

b. VADER on stopword-removed text

<img width="432" height="201" alt="vader - raw text vs stopwords 2" src="https://github.com/user-attachments/assets/255a4cf6-a4f5-4d13-af28-7660a0eb6ab3" />

c. Naive Bayes (Bag of Words)

<img width="305" height="205" alt="Naive bayes 1" src="https://github.com/user-attachments/assets/8058618e-6dc8-48ae-9a4d-2af05794e9b3" />

d. Naive Bayes (TF-IDF)

<img width="317" height="207" alt="Naive bayes 2" src="https://github.com/user-attachments/assets/68a1ff94-d8d0-49fb-b6ce-cc1acf82cff4" />

e. Logistic Regression (Bag of Words)

<img width="330" height="204" alt="logistic regression 1" src="https://github.com/user-attachments/assets/98076f49-172e-4a77-8b8a-a20cb63eed6f" />

f. Logistic Regression (TF-IDF)

<img width="324" height="154" alt="logistic regression 2" src="https://github.com/user-attachments/assets/464ccb7b-b366-4244-bc9b-85c882fb6fba" />

g. Linear SVM (Bag of Words)

<img width="311" height="201" alt="linear svm 1" src="https://github.com/user-attachments/assets/817759b7-9839-47b2-990d-595862855398" />

h. Linear SVM ((TF-IDF)

<img width="337" height="207" alt="linear svm 2" src="https://github.com/user-attachments/assets/d205c7a9-565a-4ff6-9020-395a8b5ca71a" />

The accuracy vs fairness trade-off
Linear SVM on TF-IDF looks like the best model by accuracy alone (94.0%), but its classification report shows it almost never identifies a neutral review correctly (3% recall, 5% F1). Because the test set is dominated by positive reviews (66%), a model can push accuracy up simply by getting better at the majority class while giving up on the minority one, exactly what happened here. Logistic Regression on TF-IDF, by contrast, trades a little accuracy for meaningfully better neutral class balance (22% recall, 23% F1) and the best macro F1 (0.70) among the untuned models. This is why macro F1, not accuracy, was used as the selection criterion for the final tuned model

## Hyperparameter Tuning (GridSearchCV Pipeline)
Logistic Regression + TF-IDF was carried forward for hyperparameter tuning as the strongest untuned configuration by macro F1. Rather than tuning on the already fitted TF-IDF matrix which would let each cross-validation fold's held out text leak into the vectorizer's vocabulary during fit_transform, the vectorizer, the oversampler, and the classifier were combined into a single imbalanced-learn Pipeline and tuned together directly on raw (stopword-removed) text. This keeps every fold's validation text genuinely unseen during fitting.

<img width="388" height="420" alt="hyperparameter tuning" src="https://github.com/user-attachments/assets/5221ebb8-dfd1-4f51-b634-0fba709b5c86" />

The search identified C=1, min_df=3, and an ngram_range of (1,2) as the best settings, reaching a cross-validated macro F1 of 0.6866 and, on the held-out test set, an accuracy of 93% with a macro F1 of 0.71, a further improvement in neutral-class recall (27%, up from 22% for the untuned Logistic Regression + TF-IDF model) without giving up positive/negative class performance. This tuned pipeline (best_model) was carried forward as the final model for inference and deployment.

## Inference Pipeline
The final inference() function wraps the full prediction pipeline from raw text to a sentiment label using the tuned Logistic Regression + TF-IDF model. Because best_model is the entire fitted pipeline (vectorizer included), the function only needs to strip stopwords before calling predict():
 
<img width="448" height="133" alt="inference pipeline" src="https://github.com/user-attachments/assets/3a20786f-2c75-4768-b6f6-c878c50335ad" />

<img width="627" height="86" alt="inference pipeline 2" src="https://github.com/user-attachments/assets/8990c345-cc1f-4001-beb6-bdf3e580c7bb" />

## Step 12 — Model Deployment
The tuned pipeline was serialized to disk with joblib as a single artifact, since best_model already bundles the TF-IDF vectorizer, the oversampler, and the classifier, only one file is needed:

<img width="308" height="72" alt="model deployment" src="https://github.com/user-attachments/assets/959bdfa0-c208-4ba9-b16c-4a6d2bb69cfc" />

## Limitations
 
●	Neutral class detection remains weak across every model: the best macro F1 achieved is 0.71, driven almost entirely by neutral class precision/recall still sitting in the 20–30% range even after tuning, stakeholders should not treat individual neutral predictions as highly reliable.

●	Accuracy is a misleading headline metric on this dataset: Linear SVM on TF-IDF has the highest raw accuracy (94.0%) but the weakest neutral-class recall (3%) of any model, a model selected on accuracy alone would be the worst choice for catching neutral sentiment.

●	Rating-based labels are a proxy, not ground truth: mapping star ratings to sentiment assumes a 3-star review is always “neutral” in text, which is not always true.

●	Random Oversampling only duplicates existing minority class examples rather than adding new information, so neutral class training signal is still limited to the 784 original neutral reviews in the training set, just repeated.

## Recommendations & Future Work

●	Treat macro F1 (or per class recall) as the primary model selection metric going forward, not accuracy, this dataset's imbalance makes accuracy systematically favour majority class performance.

●	Collect or synthesize more genuinely neutral reviews rather than relying solely on oversampling duplicates of the existing 957 (36,706 row) or 784 (training-only) neutral examples.

●	Run negation focused inference spot-checks before sign-off.

## Conclusion
This iteration meaningfully deepens the sentiment-analysis pipeline built for AliExpress Electronics reviews: eight model/feature combinations were compared head-to-head on a test set. The strongest configuration was carried through a hyperparameter-tuning stage. The final tuned Logistic Regression + TF-IDF pipeline reaches 93% accuracy and a macro F1 of 0.71, the best balance across all three sentiment classes of any approach tested, and was packaged as a single deployable artifact.




