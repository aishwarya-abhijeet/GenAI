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

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Stores messages for continuous conversation

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
        model = genai.GenerativeModel("gemini-1.5-pro")  # ✅ Corrected model name
        response = model.generate_content(user_input)

        # Ensure response exists and is not empty
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "⚠️ Error: No response generated. Try rephrasing your input."

    except Exception as e:
        return f"⚠️ Unexpected Error: {e}"

# Streamlit UI
st.title("🧘 AI Mental Wellness Chatbot (Powered by Google Gemini 1.5 Pro)")
st.write("Hello! I'm here to help you reflect and feel lighter. Let's chat.")

# Display chat history
for message in st.session_state.chat_history:
    role, text = message  # Each message is stored as (role, text)
    if role == "user":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 AI:** {text}")

# User input field for continuous chat
user_input = st.text_input("💬 Type your message:")

if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_input))

        # Get AI response and add to chat history
        ai_response = get_ai_response(user_input)
        st.session_state.chat_history.append(("ai", ai_response))

        # Refresh page to display new messages
        st.rerun()
