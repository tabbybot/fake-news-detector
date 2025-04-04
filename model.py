import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load Data
data_fake = pd.read_csv("Fake.csv")
data_true = pd.read_csv("True.csv")

# Assign Class Labels
data_fake["class"] = 0
data_true["class"] = 1

# Merge and Shuffle
data = pd.concat([data_fake, data_true], axis=0).drop(["title", "subject", "date"], axis=1).sample(frac=1)

fake_sample = data[data["class"] == 0].sample(n=21417, random_state=42)  
real_sample = data[data["class"] == 1]  
data = pd.concat([fake_sample, real_sample])

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

print(data["class"].value_counts())

# Split Data
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["class"], test_size=0.25, random_state=42)

# TF-IDF Vectorization (Optimized)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save Model and Vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved!")
