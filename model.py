from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from vectorizer import clean_text, get_vectorizer
import download_model

# Loading and labeling datasets
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/True.csv")
print("Datasets loaded")

fake["class"] = 0
real["class"] = 1

# Drop unnecessary columns and concat
fake = fake.drop(["title", "subject", "date"], axis=1)
real = real.drop(["title", "subject", "date"], axis=1)
data = pd.concat([fake, real], axis=0).reset_index(drop=True)

# Clean text
print("Cleaning text...")
data["cleaned_text"] = data["text"].progress_apply(clean_text)
print("Text clean-up complete")

# Undersampling Fake since Fake > Real
fake_sample = data[data["class"] == 0].sample(n=21417, random_state=42)
real_sample = data[data["class"] == 1]
data = pd.concat([fake_sample, real_sample])

# Shuffling finally to prevent possible overfitting
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced and shuffled dataset ready")

# Train test split
X = data[["cleaned_text", "text"]]
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Further split for code readability and reusability
X_train_cleaned = X_train["cleaned_text"]
X_test_cleaned = X_test["cleaned_text"]
X_test_raw = X_test["text"]

# Vectorization
vectorizer = get_vectorizer()
print("Encoding training data...")
X_train_vec = vectorizer.encode(
    X_train_cleaned.tolist(),
    convert_to_tensor=False,
    batch_size=32,
    show_progress_bar=True
)
print("Training data encoded")

print("Encoding test data...")
X_test_vec = vectorizer.encode(
    X_test_cleaned.tolist(),
    convert_to_tensor=False,
    batch_size=32,
    show_progress_bar=True
)
print("Test data encoded")

# Ensemble Fitting and saving the model
print("Preparing ensemble...")
clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(n_jobs=-1)
clf3 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

ensemble = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    ('xgb', clf3)
    ], voting='soft')

print("Fitting the ensemble...")
ensemble.fit(X_train_vec, y_train)
print("Model trained")
joblib.dump(ensemble, "model.pkl")
print("Model saved for reuse")

# Save test bundle for evaluation
print("Saving test bundle...")
test_bundle = {
    "X_test_vec": X_test_vec,
    "X_test_cleaned": X_test_cleaned,
    "X_test_raw": X_test_raw,
    "y_test": y_test
}
joblib.dump(test_bundle, "test_data_bundle.pkl")
print("All done, training complete")
