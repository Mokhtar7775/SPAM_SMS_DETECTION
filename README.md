# SMS Spam Classification

## Overview

This project focuses on building a machine learning model for classifying SMS messages as either spam or legitimate (ham). The classification is based on the content of the SMS text using natural language processing techniques.

## Dataset

The dataset used for training and evaluating the model is spam.csv It contains labeled examples of SMS messages, where each message is categorized as spam or ham.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP) techniques
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Naive Bayes Classifier

## Implementation

1. **Data Loading and Exploration:** The project starts by loading the dataset and exploring its structure. Columns 'v1' and 'v2' are identified as the label and text columns, respectively.

2. **Data Preprocessing:** The labels are converted into numerical values (0 for ham, 1 for spam), and the dataset is split into training and testing sets.

3. **Feature Extraction:** TF-IDF is employed to convert the SMS text into numerical features, capturing the importance of words in the messages.

4. **Model Training:** A Naive Bayes classifier is chosen for its simplicity and effectiveness in text classification tasks. The model is trained on the TF-IDF transformed training data.

5. **Model Evaluation:** The trained model is evaluated on the testing set, and metrics such as accuracy, confusion matrix, and classification report are calculated.

6. **Example Usage:** The model is demonstrated by predicting the spam/legitimate status of new SMS messages.

