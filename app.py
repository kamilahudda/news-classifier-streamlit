import nltk
nltk.download('stopwords')
import re
import string
from nltk.corpus import stopwords
import streamlit as st
import pickle

# Load model dan vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Stopwords bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Fungsi preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Judul aplikasi
st.title("News Category Classifier")
st.subheader("Enter a news headline to predict its category")

# Input dari user
user_input = st.text_input("News Title")

if user_input:
    clean_input = preprocess(user_input)
    X_input = vectorizer.transform([clean_input])
    prediction = model.predict(X_input)[0]
    st.success(f"Predicted Category: **{prediction}**")