
# Sentiment_Analysis_for_Customer_Feedback

![image](https://github.com/user-attachments/assets/dd4e8314-2587-49e9-a408-9ad6501d4a48)

## Table of Cotents

- [AliExpress Review Sentiment Classifier Live Demo](#AliExpress-Review-Sentiment-Classifier-Live-Demo)
- [Business Introduction](#Business-Introduction)
- [Business Problem](#Business-Problem)
- [Aim of Project](#Aim-of-Project)
- [Tech stack](#Tech-stack)
- [Data Collection](#Data-Collection)
- [Text Processing](#Text-Processing)
- [Modelling](#Modelling)
- [Model Evaluation](#Model-Evaluation)
- [Conclusion](#conclusion)

## AliExpress Review Sentiment Classifier Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentimentanalysisforcustomerfeedback-lumppzfgtqszdmt2fngwhd.streamlit.app/)

A sentiment analysis tool that classifies AliExpress electronics reviews as positive, negative, or neutral, with support for single reviews and batch (CSV uploads).

## Business Introduction
TechTrends E-commerce Solutions is a leading name in the e-commerce industry, Operate as a mid-sized company with a growing customer base. Specializing in a wide range of online product, the company places a strong emphasis on ensuring customer satisfaction. Renowned for its online presence, TechTrends actively engages with its customers, receiving a substantial volume of valuable reviews and feedback across various platforms. They face a challenge of efficiently processing and deriving valuable insight from the influx of customers reviews they receive daily. They want to develop a model to help them effectively classify customers feedback into positive and negative categories.

TechTrends E-commerce Solutions boast an impressive online presence, they have garnered over 500,000 loyal customers. This trust has translated into an astounding 98% customer satisfaction rate, and they continue to receive an average of 10,000 positive revies and feedback each month across various platforms. These statistics not only reflect their unwavering dedication to providing quality service but also underscore their position as a market leader in the e-commerce sector.

## Business Problem
TechTrends E-commerce Solutions, although a thriving player in the e-commerce sector, faces the challenge of efficiently processing and deriving valuable insights from substantial influx of customer feedback and reviews it receives daily. The company’s overarching goal is to enhance customer satisfaction and continually refine its product and service offerings. To achieve this, TechTrends E-commerce Solutions seeks to address the following specific business problem.

a.	Managing feedback overload: The company struggles with managing the sheer volume of customer feedback effectively. The influx of reviews from its extensive customer base is overwhelming and requires a streamlined approach to handle efficiently.

b.	Improving sentiment classification: TechTrends aims to enhance the accuracy of sentiment classification for feedback. Currently, the categorization of feedback into positive, negative, or neutral sentiments is not as precise as desired. Improving classification process is crucial for extracting more meaning insights.

c.	Resource allocation optimization: Manual analysis of customer feedback is resource-intensive and time-consuming. TechTrends E-commerce Solutions seeks to reduce these manual efforts through automation to allocate resources more effectively and efficiently.

## Aim of Project
My objectives can be summarized into key areas. Firstly, I aim to develop a sentiment analysis model using NLP techniques, allowing me to effectively classify customer feedback into positive and negative categories.

Secondly, I focus on implementation and performance comparison of sentiment analysis tools like NLTK and VADER.

In tandem, I aim to provide real-time sentiment analysis results to relevant teams, enabling prompt action. Simultaneously, I’m committed to extracting actionable insights from sentiment data to derive improvements in customer service and product quality. Ultimately, our overarching objective is to proactively address customer concerns and feedback, thereby enhancing the overall customer experience.

## Tech stack
Programming Language – Python

Libraries:

a.	NLTK: For natural language processing and sentiment analysis

b.	VADER: For sentiment analysis.

c.	Pandas: For data analysis and manipulation.

d.	Scikit-learn: For machine learning

## Data Collection
a.	Data Loading: Loaded in the train and test dataset using the pd.read_csv() format. I also ran data cleaning on the two datasets to see if my dataset was in the right shape for analysis. 

b.	Mapping: I mapped the labels (__label__2, __label__1) into positives and negative respectively.

## Text Processing
a.  Tokenization: Created a sentence and performed tokenization on it. Tokenization is the process of splitting sentence into list of words.

b.	Stop words Removal: Removed stop words from the sentence created above. 

c.	After tokenization, the words are returned in a list format but string format. We then need return the words in list to a string using the join function.

d.	After the three steps above, I created a function which brings the three steps together to perform all steps at once.

e.	Applied the function on the train dataset and test dataset.

f.	Created Bag of Words (Bow) And Tf-Idf (Tf-Idf), fitted and transformed on the train dataset while and transformed.

## Modelling
I worked on four difference models

a. Vader on text

b. Vader on text_without_stopwords

c. MultinimialNB on tf-idf

d. MultinimialNB on bow (Bag of Words)

For the first and second model, I imported SentimentIntensityAnalyzer from nltk.sentiment.vader library and instantiated it. Then created an example text to analyze the sentiment score of the example text. Also, I created a function that takes in a sentence and returns the sentiment using analyzer to returns positive if compound score is greater than zero, else returns negative. After then I applied the function on the original text (i.e. text with stop words) and text without stop words.

For the MultinimialNB on tf-idf and bow, I imported MultinomialNB from sklearn.naive_bayes library and instantiated it for tf-idf and bow. I went ahead to fit it on train bow & train tf-idf. After training, I predicted on test bow & test tf-idf.

## Model Evaluation
I evaluated all the four models using the classification report. The accuracy of each model is:

a.	Vader on text – 72%

b.	Vader on text_without_stopwords - 68%

c.	MultinimialNB on tf-idf – 84%

d.	MultinimialNB on bow (Bag of Words) - 85%

## Conclusion:
In conclusion, the best performing model is MultinimialNB on Bag of Words (BOW).







