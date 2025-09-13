# 📧 Spam Mail Prediction

A machine learning project that classifies emails as **Spam** or **Not Spam** using natural language processing and classification algorithms.

## 🚀 Project Overview

Spam emails are a common issue for individuals and organizations. This project aims to build a model that can accurately predict whether an email is spam or not, based on its content.

The project involves:

- Data preprocessing and cleaning
- Feature extraction using NLP techniques
- Training multiple classification models
- Evaluating model performance
- Deploying the final model (optional)

## 🧰 Tech Stack

- Python 3.x
- Google colab
- Pandas, NumPy
- Scikit-learn
- NLTK / SpaCy (for NLP)
- Jupyter Notebook / VS Code
- Matplotlib / Seaborn (for visualizations)

## 🗂️ Dataset

The dataset used is the **mail_data.csv** 

- Contains labeled email/sms messages
- Two classes: `spam` and `ham` (not spam)
- Text data used for model training

## 🛠️ Features & Workflow

1. **Data Preprocessing**  
   - Text normalization (lowercasing, removing punctuation)
   - Stopwords removal
   - Lemmatization or stemming

2. **Feature Engineering**  
   - Bag of Words / TF-IDF vectorization

3. **Model Training**  
   - Algorithms tested:
     - Naive Bayes
     - Logistic Regression
     - Support Vector Machines
     - Random Forest (optional)
   - Evaluation using Accuracy, Precision, Recall, F1-Score

4. **Model Selection**  
   - Best performing model chosen based on metrics

5. **(Optional) Deployment**  
   - Web interface using Flask / Streamlit

## 📊 Results
- Achieved **96% accuracy** on the test set
- Confusion matrix and classification report included in the notebook

