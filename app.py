import openai
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Securely Load API Key
api_key = st.secrets.get("openai_api_key")  # Use .get() to avoid KeyError
if not api_key:
    st.error("\u26a0\ufe0f OpenAI API key not found! Please set it in Streamlit secrets.")
    st.stop()

# Initialize OpenAI client
openai.api_key = api_key

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "\U0001F60A Positive"
    elif scores['compound'] <= -0.05:
        return "\U0001F61E Negative"
    else:
        return "\U0001F610 Neutral"

# Function to generate AI response
def get_ai_response(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_input}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"\u26a0\ufe0f Error generating response: {e}"

# Streamlit UI
st.title("\U0001F9D8 AI Mental Wellness Chatbot")
st.write("Hello! I'm here to help you reflect and feel lighter. Type your thoughts below.")

user_input = st.text_area("\U0001F4AC What's on your mind?", "")

if st.button("Get Support"):
    if user_input:
        ai_response = get_ai_response(user_input)
        sentiment = analyze_sentiment(user_input)
        st.write(f"**AI Response:** {ai_response}")
        st.write(f"**Mood Analysis:** {sentiment}")
    else:
        st.warning("Please enter some text before submitting.")

st.write("\U0001F4A1 Try writing about your emotions, challenges, or thoughts. I'll help you process them.")
