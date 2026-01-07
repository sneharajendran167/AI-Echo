import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Page config
st.set_page_config(
    page_title="AI Echo - Sentiment Analysis",
    layout="wide"
)

st.title("🤖 AI Echo - Sentiment Analysis Dashboard")
st.write("Interactive analysis of customer reviews using NLP")

# Load dataset

df=pd.read_csv("AI Echo.csv")


# Create Sentiment Column

def label_sentiment(r):
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(label_sentiment)

# Create review length
df["review_length"] = df["review"].astype(str).apply(len)

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")


# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Select Section",
    ["1)Overall Sentiment", "2)Sentiment vs Rating", "3)Keywords by Sentiment",
     "4)Sentiment Over Time", "5)Verified Users", "6)Review Length",
     "7)Location-wise Sentiment", "8)Platform-wise Sentiment",
     "9)Version-wise Sentiment", "10)Negative Feedback Themes"]
)


# 1. Overall Sentiment

if section == "1)Overall Sentiment":
    st.header("1) Overall Sentiment of User Reviews")

    sentiment_counts = df["sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index,
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Overall Sentiment Distribution")
    st.pyplot(fig)


# 2. Sentiment vs Rating

elif section == "2)Sentiment vs Rating":
    st.header("2) Sentiment Variation by Rating")

    fig, ax = plt.subplots()
    sns.countplot(x="rating", hue="sentiment", data=df, ax=ax)
    ax.set_title("Rating vs Sentiment")
    st.pyplot(fig)


# 3. Keywords by Sentiment

elif section == "3)Keywords by Sentiment":
    st.header("3) Keywords Associated with Each Sentiment")

    sentiment_choice = st.selectbox(
        "Select Sentiment",
        ["Positive", "Neutral", "Negative"]
    )

    text = " ".join(df[df["sentiment"] == sentiment_choice]["review"].astype(str))

    wc = WordCloud(background_color="white", max_words=200)
    fig, ax = plt.subplots()
    ax.imshow(wc.generate(text))
    ax.axis("off")
    ax.set_title(f"{sentiment_choice} Review Keywords")
    st.pyplot(fig)


# 4. Sentiment Over Time

elif section == "4)Sentiment Over Time":
    st.header("4) Sentiment Change Over Time")

    trend = df.groupby([df["date"].dt.to_period("M"), "sentiment"]).size().unstack()

    fig, ax = plt.subplots(figsize=(8,4))
    trend.plot(ax=ax)
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)


# 5. Verified Users

elif section == "5)Verified Users":
    st.header("5) Verified vs Non-Verified User Sentiment")

    fig, ax = plt.subplots()
    sns.countplot(x="verified_purchase", hue="sentiment", data=df, ax=ax)
    ax.set_title("Sentiment by Verification Status")
    st.pyplot(fig)


# 6. Review Length

elif section == "6)Review Length":
    st.header("6) Review Length vs Sentiment")

    fig, ax = plt.subplots()
    sns.boxplot(x="sentiment", y="review_length", data=df, ax=ax)
    ax.set_title("Review Length by Sentiment")
    st.pyplot(fig)


# 7. Location-wise Sentiment

elif section == "7)Location-wise Sentiment":
    st.header("7) Location-wise Sentiment")

    location_sentiment = (
        df.groupby("location")["sentiment"]
        .value_counts()
        .unstack()
        .fillna(0)
    )

    st.dataframe(location_sentiment)


# 8. Platform-wise Sentiment

elif section == "8)Platform-wise Sentiment":
    st.header("8) Sentiment Across Platforms")

    fig, ax = plt.subplots()
    sns.countplot(x="platform", hue="sentiment", data=df, ax=ax)
    ax.set_title("Platform-wise Sentiment")
    st.pyplot(fig)


# 9. Version-wise Sentiment

elif section == "9)Version-wise Sentiment":
    st.header("9) Sentiment by ChatGPT Version")

    fig, ax = plt.subplots()
    sns.countplot(x="version", hue="sentiment", data=df, ax=ax)
    ax.set_title("Version-wise Sentiment")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# 10. Negative Feedback Themes

elif section == "10)Negative Feedback Themes":
    st.header("10) Common Negative Feedback Themes")

    negative_text = " ".join(
        df[df["sentiment"] == "Negative"]["review"].astype(str)
    )

    wc = WordCloud(background_color="black", max_words=150)
    fig, ax = plt.subplots()
    ax.imshow(wc.generate(negative_text))
    ax.axis("off")
    ax.set_title("Negative Review Keywords")
    st.pyplot(fig)

st.markdown("---")
st.markdown("📌 **AI Echo - Sentiment Analysis Project** | NLP & Machine Learning")
