import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


# NLTK setup

nltk.download('stopwords')
nltk.download('wordnet')

# Page Config

st.set_page_config(page_title="AI Echo Dashboard", layout="wide")
st.title("🧠 AI Echo: Sentiment Analysis Dashboard")
st.markdown("Analyze ChatGPT user reviews using NLP & Machine Learning")


# Load Data

@st.cache_data
def load_data():
    df = pd.read_csv("AI Echo.csv")

    df.columns = [
        'date','title','review','rating','username',
        'helpful_votes','review_length','platform',
        'language','location','version','verified_purchase'
    ]

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    return df

df = load_data()


# NLP Cleaning

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

df['combined_text'] = df['title'].fillna('') + " " + df['review'].fillna('')
df['clean_text'] = df['combined_text'].apply(clean_text)


# Load Model & Predict Sentiment

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

X_all = vectorizer.transform(df['clean_text'])
df['sentiment'] = model.predict(X_all)


# Sidebar Filters

st.sidebar.header("📌 Filters")

platform_filter = st.sidebar.multiselect(
    "Platform",
    df['platform'].unique(),
    default=df['platform'].unique()
)

version_filter = st.sidebar.multiselect(
    "Version",
    df['version'].unique(),
    default=df['version'].unique()
)

df = df[(df['platform'].isin(platform_filter)) & (df['version'].isin(version_filter))]


# 1️ Overall Sentiment

st.subheader("1) Overall Sentiment of User Reviews")

fig, ax = plt.subplots()
df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
ax.set_ylabel("")
st.pyplot(fig)


# 2️ Sentiment vs Rating

st.subheader("2) Sentiment Variation by Rating")

fig, ax = plt.subplots()
sns.countplot(x='rating', hue='sentiment', data=df, ax=ax)
st.pyplot(fig)


# 3 Keywords per Sentiment

st.subheader("3) Keywords Associated with Each Sentiment")

for s in df['sentiment'].unique():
    text = " ".join(df[df['sentiment'] == s]['clean_text'])
    wc = WordCloud(width=600, height=300, background_color='white')

    fig, ax = plt.subplots()
    ax.imshow(wc.generate(text))
    ax.axis("off")
    ax.set_title(s)
    st.pyplot(fig)


# 4️ Sentiment Over Time

st.subheader("4) Sentiment Change Over Time")

trend = df.groupby([df['date'].dt.to_period('M'), 'sentiment']).size().unstack()

fig, ax = plt.subplots()
trend.plot(ax=ax)
ax.set_ylabel("Review Count")
st.pyplot(fig)


# 5 Verified Users

st.subheader("5) Sentiment by Verified Purchase")

fig, ax = plt.subplots()
sns.countplot(x='verified_purchase', hue='sentiment', data=df, ax=ax)
st.pyplot(fig)


# 6 Review Length vs Sentiment

st.subheader("6) Review Length vs Sentiment")

fig, ax = plt.subplots()
sns.boxplot(x='sentiment', y='review_length', data=df, ax=ax)
st.pyplot(fig)


# 7 Sentiment by Location

st.subheader("7) Sentiment by Location (Top 10)")

top_locations = df['location'].value_counts().head(10).index
df_loc = df[df['location'].isin(top_locations)]

fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='location', hue='sentiment', data=df_loc, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# 8 Platform Sentiment

st.subheader("8) Sentiment Across Platforms")

fig, ax = plt.subplots()
sns.countplot(x='platform', hue='sentiment', data=df, ax=ax)
st.pyplot(fig)


# 9 Version Sentiment

st.subheader("9)Sentiment by ChatGPT Version")

fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='version', hue='sentiment', data=df, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)


#10 Negative Feedback Themes

st.subheader("10) Common Negative Feedback Themes")

neg_text = " ".join(df[df['sentiment'] == 'Negative']['clean_text'])
wc = WordCloud(width=700, height=350, background_color='white')

fig, ax = plt.subplots()
ax.imshow(wc.generate(neg_text))
ax.axis("off")
st.pyplot(fig)


# 💬 Live Sentiment Prediction

st.header("💬 Live Sentiment Prediction")

user_input = st.text_area("Enter a review")

if st.button("Predict Sentiment"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]

    if pred == "Positive":
        st.success("✅ Positive Sentiment")
    elif pred == "Neutral":
        st.info("⚖️ Neutral Sentiment")
    else:
        st.error("❌ Negative Sentiment")

