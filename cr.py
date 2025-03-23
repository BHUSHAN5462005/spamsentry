import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv", encoding="latin-1")
    data = data.iloc[:, [0, 1]]  # Selecting only label and message columns
    data.columns = ["label", "message"]
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    return data

data = load_data()

# Train TF-IDF Vectorizer with 5000 features
@st.cache_resource
def train_vectorizer():
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Increased features
    X_tfidf = vectorizer.fit_transform(data["message"])  # Fit on full dataset
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    return vectorizer

vectorizer = train_vectorizer()

# Train Naive Bayes Model
@st.cache_resource
def train_model():
    X_tfidf = vectorizer.transform(data["message"])
    model = MultinomialNB()
    model.fit(X_tfidf, data["label"])
    joblib.dump(model, "spam_classifier.pkl")
    return model

model = train_model()

# Load model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to apply keyword-based boosting
def keyword_boost(text):
    spam_keywords = {
        "free": 0.15, "win": 0.2, "winner": 0.2, "congratulations": 0.15, 
        "claim": 0.2, "click": 0.15, "urgent": 0.25, "lottery": 0.3,
        "transfer": 0.35, "bank account": 0.35, "confidential": 0.35, 
        "risk-free": 0.4, "prince": 0.5, "Nigeria": 0.5,
        "work from home": 0.3, "earn": 0.2, "per month": 0.25, 
        "apply now": 0.3, "hiring": 0.2, "no experience": 0.25,
        # Increased weights for financial/spam terms
        "double the money": 0.7, "invest": 0.6, "investment": 0.6, 
        "money back": 0.25, "fast cash": 0.7, "guaranteed": 0.5, 
        "limited offer": 0.4, "phone number": 0.6
    }
    boost = sum(weight for word, weight in spam_keywords.items() if re.search(rf"\b{word}\b", text, re.IGNORECASE))

    # Detect phone numbers (7 or more digits)
    if re.search(r"\b\d{7,}\b", text):  
        boost += 0.6  # Stronger impact for phone numbers

    return boost

# Streamlit UI
st.title("📧 SPAMSENTRY ")
st.write("Enter an email message below to check if it's spam or not.")

# Input text box
user_input = st.text_area("Enter Email Message Here:")

if st.button("Classify"):
    if user_input.strip():
        # Transform input text
        user_input_tfidf = vectorizer.transform([user_input])
        
        # Predict probability
        spam_prob = model.predict_proba(user_input_tfidf)[0][1]
        boosted_prob = spam_prob + keyword_boost(user_input)  # Apply keyword boost
        boosted_prob = min(boosted_prob, 1.0)  # Keep probability within valid range

        # Display result
        st.subheader("Result:")
        if boosted_prob >= 0.4:  # 🔥 Reduced threshold from 0.5 to 0.4
            st.error(f"🚨 This email is **Spam!** (Confidence: {boosted_prob:.2f})")
        else:
            st.success(f"✅ This email is **Ham.** (Confidence: {boosted_prob:.2f})")
    else:
        st.warning("Please enter an email message.")
