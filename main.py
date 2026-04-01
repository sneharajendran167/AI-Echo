import pandas as pd 
import streamlit as st
import pickle
import re
import string
import plotly.express as px
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt

# Load best model
model = pickle.load(open("sentiment_model1.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer1.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(
    page_title="AI Echo - Sentiment Analysis",
    layout="wide",
    page_icon="💬"
)


option = st.sidebar.selectbox(
    "🧭 Navigation",
    ("📘 Project Explanation", "📈 EDA Charts", "💡 Model Prediction")
)


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

df1=pd.read_csv("D:/AI ECHO/df_1.csv")


def get_sentiment(rating):
    rating = int(rating)
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df1['sentiment'] = df1['rating'].apply(get_sentiment)

overall_sentiment = df1['sentiment'].value_counts()

def chart_1():
    overall_sentiment = df1['sentiment'].value_counts()
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
    rating_sentiment = df1.groupby('rating')['sentiment'].value_counts().reset_index(name='count')
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

        text = ' '.join(df1[df1['sentiment'] == sentiment]['review'].astype(str))
        wc = WordCloud(width=700, height=400, background_color='black').generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')

        st.pyplot(fig)


def chart_5():
    data = df1.groupby('verified_purchase')['sentiment'].value_counts().reset_index(name='count')
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
    if 'review_length' not in df1.columns:
        df1['review_length'] = df1['review'].astype(str).apply(len)

    review_length_sentiment = (
        df1
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
        df1
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
        df1
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
        df1
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
    text = ' '.join(df1[df1['sentiment'] == 'Negative']['review'].astype(str))
    wc = WordCloud(width=700, height=400, background_color='black').generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')

    st.pyplot(fig)

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
    st.title("📊 Sentiment Analysis")

    user_input = st.text_area("Enter your review")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Enter text")
        else:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            prediction = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0] 

            # IMPROVED DECISION
            max_prob = max(proba)

            if max_prob < 0.6:
                final_pred = "neutral"
            else:
                final_pred = prediction

            # DISPLAY RESULT
            if final_pred == "positive":
                st.success("😊 Positive Sentiment")
            elif final_pred == "negative":
                st.error("😡 Negative Sentiment")
            else:
                st.info("😐 Neutral Sentiment") 