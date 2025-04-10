import traceback
import streamlit as st
import joblib
from newspaper import Article
from vectorizer import clean_text, get_vectorizer
import numpy as np

model = joblib.load("model.pkl")
vectorizer = get_vectorizer()

st.title("Fake News Detector 📰")
st.header("Fake News Detector from URL")
st.write("Note: Our project may not be able to extract news articles from certain websites due to web scraping being blocked")
st.write("tabbybot et al. 2025")
url = st.text_input("Paste a news article URL here")

if url:
    try:
        article = Article(url, language='en', browser_user_agent='Mozilla/5.0')
        article.download()
        article.parse()
        st.success("Article fetched successfully!")
        st.write("### Article Title:")
        st.write(article.title)

        st.write("### Article Text:")
        st.write(article.text[:1000] + "...")

        if st.button("Check if it's Fake"):
            cleaned_text = clean_text(article.text)
            embeddings = np.array(vectorizer.encode([cleaned_text]))

            prediction = model.predict(embeddings)[0]
            proba = model.predict_proba(embeddings)[0]

            label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
            st.subheader(f"Prediction: {label}")

            st.write(f"**Fake News Confidence:** {proba[0]*100:.2f}%")
            st.write(f"**Real News Confidence:** {proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Couldn't process the URL: {e}")
        st.code(traceback.format_exc(), language="python")

