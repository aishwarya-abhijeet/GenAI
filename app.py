import streamlit as st
import google.generativeai as genai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Securely Fetch Google Gemini API Key
gemini_api_key = st.secrets.get("gemini_api_key")

# Validate API key before proceeding
if not gemini_api_key or not gemini_api_key.startswith("AIza"):
    st.error("⚠️ Error: Google Gemini API key is missing or invalid! Please check Streamlit secrets.")
    st.stop()

# Configure Google Gemini API
genai.configure(api_key=gemini_api_key)

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "😊 Positive"
    elif scores['compound'] <= -0.05:
        return "😞 Negative"
    else:
        return "😐 Neutral"

# Function to generate AI response using Google Gemini
def get_ai_response(user_input):
    try:
        model = genai.GenerativeModel("gemini-pro-latest")  # ✅ Correct model name
        response = model.generate_content(user_input)

        # Ensure response exists and is not empty
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "⚠️ Error: No response generated. Try rephrasing your input."

    except genai.types.NotFoundError:
        return "⚠️ Error: Model not found. Your API key may not have access."
    except Exception as e:
        return f"⚠️ Unexpected Error: {e}"

# Streamlit UI
st.title("🧘 AI Mental Wellness Chatbot (Powered by Google Gemini)")
st.write("Hello! I'm here to help you reflect and feel lighter. Type your thoughts below.")

user_input = st.text_area("💬 What's on your mind?", "")

if st.button("Get Support"):
    if user_input:
        ai_response = get_ai_response(user_input)
        sentiment = analyze_sentiment(user_input)
        st.write(f"**AI Response:** {ai_response}")
        st.write(f"**Mood Analysis:** {sentiment}")
    else:
        st.warning("⚠️ Please enter some text before submitting.")

st.write("💡 Try writing about your emotions, challenges, or thoughts. I'll help you process them.")
