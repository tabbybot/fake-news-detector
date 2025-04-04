import streamlit as st
import joblib
from newspaper3k import Article

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake News Detector 📰")

st.header("📰 Fake News Detector from URL")

url = st.text_input("Paste a news article URL here")

if url:
    try:
        article = Article(url)
        article.download()
        article.parse()
        st.success("Article fetched successfully!")
        st.write("### Article Title:")
        st.write(article.title)

        st.write("### Article Text:")
        st.write(article.text[:1000] + "...")  # Don't show everything if it's long

        if st.button("Check if it's Fake"):
            processed = vectorizer.transform([article.text])
            prediction = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0]  # Get confidence scores

            label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
            st.subheader(f"Prediction: {label}")

            # Show both confidence scores
            st.write(f"**Fake News Confidence:** {proba[0]*100:.2f}%")
            st.write(f"**Real News Confidence:** {proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Couldn't process the URL: {e}")

