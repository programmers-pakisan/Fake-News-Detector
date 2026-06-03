import streamlit as st
import pickle

# Load Model and TF-IDF
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

# App Title
st.title("📰 Fake News Detector")

st.write("Enter a news article below to check if it is Real or Fake.")

# Input Box
news = st.text_area("Enter News Article")

# Button
if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")

    else:
        # Convert text into TF-IDF features
        vector = tfidf.transform([news])

        # Prediction
        prediction = model.predict(vector)

        # Result
      if prediction[0] == 1:
    st.error("⚠️ Fake News")
else:
    st.success("✅ Real News")
