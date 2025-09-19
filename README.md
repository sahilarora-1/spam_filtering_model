# 📧 Email Spam Detection Model

This repository contains a trained machine learning model to classify emails as spam or non-spam (ham). It is designed to be integrated into email automation systems, bots, or standalone applications.

# Features

Predicts whether an email is spam or not.
High accuracy (98–99%) on test data.
Lightweight and easy to integrate.
# Preprocessing included: 
cleans HTML, special characters, and lowercases text.

Uses TF-IDF vectorization and Multinomial Naive Bayes.

# Dataset

The model was trained on a labeled email dataset consisting of 10,000 emails data
Column	Description
Body	Email text content
Label	Target label: 0 = non-spam, 1 = spam

# Installation

clone the repo:https://github.com/sahilarora-1/spam_filtering_model.git

git clone https://github.com/sahilarora-1/spam_filtering_model.git

go to the folder
cd spam_filtering_model

Install dependencies:

pip install -r requirements.txt


Dependencies include:

pandas

scikit-learn

numpy

pickle (built-in)

Usage
1. Load the model and vectorizer
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

2. Preprocess and predict
import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)   # remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special characters
    text = text.lower()
    return text

email_text = "Congratulations! You have won a prize."
cleaned_text = clean_text(email_text)
vectorized_text = vectorizer.transform([cleaned_text])

prediction = model.predict(vectorized_text)[0]
print("Spam" if prediction == 1 else "Not Spam")

Model Performance
Metric	Score
Accuracy	0.9855
Precision	0.98–0.99
Recall	0.98–0.99
F1-Score	0.99

Balanced performance for both spam and non-spam emails.

Ready for real-world usage and integration.

Contributing

Feel free to:

Improve preprocessing

Train on larger datasets

Integrate into email automation systems
