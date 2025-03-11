import openai
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Fetch OpenAI API key securely
openai_api_key = st.secrets["openai_api_key"]

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "😊 Positive"
    elif scores['compound'] <= -0.05:
        return "😞 Negative"
    else:
        return "😐 Neutral"

# Function to generate AI response
def get_ai_response(user_input):
    client = openai.OpenAI(api_key=openai_api_key)  # Pass API key
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("🧘 AI Mental Wellness Chatbot")
st.write("Hello! I'm here to help you reflect and feel lighter. Type your thoughts below.")

user_input = st.text_area("💬 What's on your mind?", "")

if st.button("Get Support"):
    if user_input:
        ai_response = get_ai_response(user_input)
        sentiment = analyze_sentiment(user_input)
        st.write(f"**AI Response:** {ai_response}")
        st.write(f"**Mood Analysis:** {sentiment}")
    else:
        st.warning("Please enter some text before submitting.")

st.write("💡 Try writing about your emotions, challenges, or thoughts. I'll help you process them.")
