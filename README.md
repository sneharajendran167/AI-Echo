**🧠 AI Echo – Sentiment Analysis Project**

AI Echo – Your Smartest Conversational Partner

**📌 Project Overview**

AI Echo is a Sentiment Analysis project that analyzes user reviews using Natural Language Processing (NLP) and Machine Learning to classify sentiments into Positive, Neutral, and Negative categories.
The project helps understand user feedback, improve product quality, and enhance user experience through data-driven insights.

**🎯 Project Objectives**

Analyze customer reviews and ratings

Clean and preprocess textual data

Perform Exploratory Data Analysis (EDA)

Build a sentiment classification model

Evaluate model performance

Identify important keywords influencing sentiment

Deploy results using a Streamlit dashboard

**🗂️ Dataset Details**

Dataset Name: AI Echo.csv

Description: User reviews and ratings data

**Key Columns:**

review – Textual user feedback

rating – User rating (1 to 5)

platform – Platform used

version – App version

location – User location

verified_purchase – Verification status

date – Review date

**Sentiment Labeling Logic:**

Rating ≥ 4 → Positive

Rating = 3 → Neutral

Rating ≤ 2 → Negative

**🛠️ Technologies & Tools Used**

Python

Pandas, NumPy

Scikit-learn

TF-IDF Vectorizer

Logistic Regression

Random Forest Classification

Matplotlib & Seaborn

Streamlit

**🔄 Project Workflow**

1️⃣ Data Loading

2️⃣ Data Cleaning & Text Preprocessing

3️⃣ Exploratory Data Analysis (EDA)

4️⃣ Feature Extraction using TF-IDF

5️⃣ Model Building

6️⃣ Model Evaluation

7️⃣ Streamlit Dashboard Deployment


**🤖 Model Description**

Two machine learning models were implemented and compared for sentiment classification:

🔹 Logistic Regression

Simple and interpretable model

Performs efficiently with TF-IDF features

Provides clear feature importance through coefficients

🔹 Random Forest Classifier

Ensemble-based model using multiple decision trees

Captures non-linear patterns in text data

Reduces overfitting through bagging

Used for performance comparison

👉 Logistic Regression was selected as the final model due to its interpretability and comparable performance.


**🔍 Feature Importance & Key Insights**
Positive Sentiment:

Easy

Helpful

Fast

Accurate

User-friendly

Negative Sentiment:

Bugs

Issues

Crashes

Slow performance

Neutral Sentiment:

Average

Okay

Decent

👉 These insights help identify strengths and areas for improvement.

**🌐 Streamlit Dashboard Features**

Overall sentiment distribution

Sentiment vs rating analysis

Platform-wise and version-wise performance

Word cloud visualization

Time-based sentiment trends

Verified vs non-verified user analysis

Real-time sentiment prediction for new reviews

**▶ How to Run the Project**
Install required libraries
pip install streamlit pandas scikit-learn matplotlib seaborn

Run the Streamlit app
streamlit run aiecho.py

**Screenshots**
<img width="1920" height="1080" alt="Screenshot (38)" src="https://github.com/user-attachments/assets/c851cfe2-f41d-454e-a9f9-1132083aa884" />

<img width="1920" height="1080" alt="Screenshot (39)" src="https://github.com/user-attachments/assets/af51e537-11d7-475e-8667-17c31b48ef6d" />

<img width="1919" height="1009" alt="Screenshot 2026-02-13 174747" src="https://github.com/user-attachments/assets/1633177b-321c-43ff-8960-355da4fe2e55" />



**📁 Project Files**

aiecho.ipynb – Data analysis, model building, and evaluation

aiecho.py – Streamlit dashboard application

AI Echo.csv – Dataset

README.md – Project documentation

**🏁 Conclusion**

AI Echo demonstrates a complete end-to-end NLP pipeline, from raw text processing to model deployment.
The project successfully combines machine learning, data visualization, and business insights to understand user sentiment effectively.

@ Contact: 📧 Email: sneharaje167@gmail.com

🌐 LinkedIn: https://www.linkedin.com/in/sneha-rajendiran-2427651b7
