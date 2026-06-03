import streamlit as st
import pickle
import sklearn

with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.title("📰 Fake News Detector")

news = st.text_area("Enter News Article")

if st.button("Check News"):

    if news.strip() != "":

        vector = tfidf.transform([news])

        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.error("⚠️ Fake News")
        else:
            st.success("✅ Real News")
