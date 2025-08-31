# SENTIMENT ANALYSIS - CodTech Internship Task 4

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# NLP + ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
# Example dataset: Movie reviews (replace with your dataset e.g., tweets.csv, reviews.csv)
data = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")

print("Dataset shape:", data.shape)
print(data.head())

# Step 3: Data Preprocessing
import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)      # remove mentions
    text = re.sub(r'#\w+', '', text)      # remove hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

data['cleaned_text'] = data['tweet'].apply(clean_text)

# Step 4: Features and Labels
X = data['cleaned_text']
y = data['label']   # 0 = Negative, 1 = Positive

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 7: Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test_vec)

# Step 9: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10: Test with Custom Input
sample_texts = ["I love this internship!", "This project is very boring and bad."]
sample_vec = vectorizer.transform(sample_texts)
predictions = model.predict(sample_vec)

for text, label in zip(sample_texts, predictions):
    print(f"Text: {text} -> Sentiment: {'Positive' if label==1 else 'Negative'}")