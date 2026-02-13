# ======================================================
# AI Echo : Your Smartest Conversational Partner
# Sentiment Analysis Streamlit App
# ======================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import joblib

# ======================================================
# 1️⃣ MODEL LOADING
# ======================================================

@st.cache_resource
def load_model():
    model = joblib.load(r"D:\sentiment analysis project\logistic.pkl")
    return model

model = load_model()


# ======================================================
# 2️⃣ LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    return pd.read_csv(
    r"D:\sentiment analysis project\AI Echo.csv"
)


review_df = load_data()

def predict_sentiment_bulk(df):
    df['sentiment'] = model.predict(df['review'].astype(str))
    return df

review_df = predict_sentiment_bulk(review_df)

overall_sentiment = review_df['sentiment'].value_counts()


# ======================================================
# 3️⃣ STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Echo - Sentiment Analysis",
    layout="wide",
    page_icon="💬"
)

# ======================================================
# 4️⃣ SIDEBAR NAVIGATION
# ======================================================
option = st.sidebar.selectbox(
    "🧭 Navigation",
    ("📘 Project Explanation", "📈 EDA Charts", "💡 Model Prediction")
)

# ======================================================
# 5️⃣ DROPDOWN FOR CHARTS
# ======================================================
chart_options = {
    "Overall sentiment of user reviews": "chart1",
    "Sentiment variation by rating": "chart2",
    "keywords or phrases are most associated with each sentiment class" : "chart_3",
    "verified users tend to leave more positive or negative reviews": "chart_5",
    "longer reviews more likely to be negative or positive": "chart_6",
    "locations show the most positive or negative sentiment": "chart_7",
    "difference in sentiment across platforms (Web vs Mobile)": "chart_8",
    "ChatGPT versions are associated with higher/lower sentiment": "chart_9",
    "the most common negative feedback themes": "chart_10"
}

# ======================================================
# 6️⃣ EDA CHART FUNCTIONS
# ======================================================

def chart_1():
    overall_sentiment = review_df['sentiment'].value_counts()
    st.write(overall_sentiment)

    fig = px.bar(
        x=overall_sentiment.index,
        y=overall_sentiment.values,
        color=overall_sentiment.values,
        title='Overall Sentiment of User Reviews'
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def chart_2():
    rating_sentiment = review_df.groupby('rating')['sentiment'].value_counts().reset_index(name='count')
    st.write(rating_sentiment.head())

    fig = px.bar(
            rating_sentiment,
            x='rating',
            y='count',
            color='sentiment',
            title='Sentiment by Rating'
        )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def chart_3():
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        st.subheader(f"{sentiment} Reviews")

        text = ' '.join(review_df[review_df['sentiment'] == sentiment]['review'].astype(str))
        wc = WordCloud(width=700, height=400, background_color='black').generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')

        st.pyplot(fig)


def chart_5():
    data = review_df.groupby('verified_purchase')['sentiment'].value_counts().reset_index(name='count')
    st.write(data.head())

    fig = px.bar(
        data,
        x='verified_purchase',
        y='count',
        color='sentiment',
        barmode='stack',
        title='Sentiment by Verified Purchase'
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

def chart_6():
    # Create review length if not already present
    if 'review_length' not in review_df.columns:
        review_df['review_length'] = review_df['review'].astype(str).apply(len)

    review_length_sentiment = (
        review_df
        .groupby('sentiment')['review_length']
        .mean()
        .reset_index()
    )

    fig = px.bar(
        review_length_sentiment,
        x='sentiment',
        y='review_length',
        color='sentiment',
        title='Average Review Length by Sentiment'
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def chart_7():
    location_sentiment = (
        review_df
        .groupby('location')['sentiment']
        .value_counts()
        .reset_index(name='count')
    )

    fig = px.bar(
        location_sentiment,
        x='location',
        y='count',
        color='sentiment',
        barmode='stack',
        title='Sentiment Distribution by Location'
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)


def chart_8():
    platform_sentiment = (
        review_df
        .groupby('platform')['sentiment']
        .value_counts()
        .reset_index(name='count')
    )

    fig = px.bar(
        platform_sentiment,
        x='platform',
        y='count',
        color='sentiment',
        barmode='stack',
        title='Sentiment Distribution by Platform'
    )

    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def chart_9():
    version_sentiment = (
        review_df
        .groupby('version')['sentiment']
        .value_counts()
        .reset_index(name='count')
    )

    fig = px.bar(
        version_sentiment,
        x='version',
        y='count',
        color='sentiment',
        barmode='stack',
        title='Sentiment Distribution by ChatGPT Version'
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig, use_container_width=True)


def chart_10():
    text = ' '.join(review_df[review_df['sentiment'] == 'Negative']['review'].astype(str))
    wc = WordCloud(width=700, height=400, background_color='black').generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')

    st.pyplot(fig)

# ======================================================
# 7️⃣ MAIN PAGE LOGIC
# ======================================================

if option == "📘 Project Explanation":
    st.markdown(
        '<h1 style="color:#00c8ff; font-family:Arial; text-align:center;">💬 AI Echo: Your Smartest Conversational Partner</h1>',
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("📘 Project Overview")

    st.markdown("""
    **Objective:**  
    AI Echo is an intelligent system that analyzes user reviews to predict sentiment and uncover insights.  
    The project demonstrates how Natural Language Processing (NLP) and Machine Learning (ML) can transform textual feedback into actionable information.

    **Key Features:**

    **1️⃣ Sentiment Analysis:**  
    - Classifies user reviews into Positive 😄, Neutral 😐, and Negative 😞 sentiments.  
    - Provides probability scores to indicate prediction confidence.

    **2️⃣ Exploratory Data Analysis (EDA):**  
    - Visualizes overall sentiment distribution.  
    - Analyzes sentiment trends across ratings, platforms, locations, and ChatGPT versions.  
    - Identifies if verified users tend to leave more positive reviews.  
    - Examines review length patterns to see if longer reviews correlate with sentiment.

    **3️⃣ Keyword & Feedback Analysis:**  
    - Generates Word Clouds for each sentiment class to highlight most frequent words.  
    - Identifies common negative feedback themes for actionable insights.

    **4️⃣ Interactive UI with Streamlit:**  
    - Sidebar navigation for switching between “Project Explanation” and “Model Prediction.”  
    - Dropdown selection for viewing EDA charts.  
    - Input box for users to enter custom reviews and get sentiment predictions instantly.  
    - Dynamic visualization using Plotly and WordCloud integrated into the dashboard.
    """)

    st.markdown("""
    **5️⃣ Tech Stack Used:**<br>

    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="30"/> Python (pandas, matplotlib, Plotly, WordCloud)<br>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/streamlit/streamlit-original.svg" width="30"/> Streamlit for interactive web interface<br>
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="30"/> scikit-learn for Machine Learning<br>
    📦 Joblib for model serialization<br>
    🧠 NLTK / NLP techniques for preprocessing and feature extraction
    """, unsafe_allow_html=True) 

    st.markdown("""
        **Impact:**  
        This project enables companies or developers to quickly understand user feedback, measure product satisfaction, and identify recurring pain points.  
        It’s designed to be visually appealing, interactive, and easily extendable for real-world applications.
        """)


elif option == "📈 EDA Charts":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Select a chart to explore the sentiment insights!")

    selected_chart = st.selectbox(
        "Choose a chart:",
        list(chart_options.keys())
    )

    if selected_chart == "Overall sentiment of user reviews":
        chart_1()
    elif selected_chart == "Sentiment variation by rating":
        chart_2()
    elif selected_chart == "keywords or phrases are most associated with each sentiment class":
        chart_3()
    elif selected_chart == "verified users tend to leave more positive or negative reviews":
        chart_5()
    elif selected_chart == "longer reviews more likely to be negative or positive":
        chart_6()
    elif selected_chart == "locations show the most positive or negative sentiment":
        chart_7()
    elif selected_chart == "difference in sentiment across platforms (Web vs Mobile)":
        chart_8()
    elif selected_chart == "ChatGPT versions are associated with higher/lower sentiment":
        chart_9()
    elif selected_chart == "the most common negative feedback themes":
        chart_10()

elif option == "💡 Model Prediction":
    st.title("🧠 Sentiment Prediction")
    st.markdown("Enter your review and let AI predict the sentiment!")

    user_input = st.text_area("✍️ Type your review below:")

    if st.button("🔍 Predict Sentiment"):
        if user_input.strip():
            pred = model.predict([user_input])[0]
            pred_proba = model.predict_proba([user_input])[0]
            st.markdown(f"### **Predicted Sentiment:** {pred}")
            st.success(f"Confidence: {max(pred_proba)*100:.2f}%")
        else:
            st.warning("⚠️ Please enter a review text!")
