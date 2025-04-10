import re
import string
import joblib
import torch
from sentence_transformers import SentenceTransformer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def get_vectorizer(model_name="local_model"):
    print("Loading transformer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    model = SentenceTransformer(model_name, device=device)
    print("Loaded")
    return model
