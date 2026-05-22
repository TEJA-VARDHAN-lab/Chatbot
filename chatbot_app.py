import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page layout and title
st.set_page_config(page_title="Customer Service Chatbot", page_icon="💬", layout="wide")

# ==========================================
# 1. KNOWLEDGE BASE (FAQs)
# ==========================================
# A dictionary of predefined customer queries and corresponding answers.
FAQ_DATA = {
    "What are your business hours?": 
        "Our customer support and store hours are Monday to Friday, 9:00 AM to 6:00 PM EST. We are closed on weekends and major holidays.",
    
    "How can I track my package?": 
        "Once your order ships, we send a tracking link to your registered email. You can also view the status by logging into your account under 'My Orders'.",
    
    "What is your refund or return policy?": 
        "We offer a 30-day hassle-free return policy. Items must be unused, in their original packaging, and accompanied by the receipt.",
    
    "Do you offer international shipping?": 
        "Yes, we ship to over 50 countries worldwide! International shipping rates and delivery times are calculated at checkout based on your location.",
    
    "How do I cancel my order?": 
        "You can cancel your order within 1 hour of placing it. Please go to your order history and click 'Cancel Order'. After 1 hour, the order is processed and cannot be cancelled.",
    
    "How can I contact a human agent?": 
        "If you need further assistance, you can email our support team at support@example.com or call us at 1-800-555-0199 during business hours.",
        
    "What payment methods do you accept?": 
        "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay."
}

# Convert FAQs to a list for vectorization
faq_questions = list(FAQ_DATA.keys())
faq_answers = list(FAQ_DATA.values())

# ==========================================
# 2. NLP ENGINE (TF-IDF & Similarity Matcher)
# ==========================================
def get_bot_response(user_query, threshold=0.3):
    """
    Matches the user's query against the FAQ list using TF-IDF and Cosine Similarity.
    If the best match similarity is below the threshold, it triggers a fallback.
    """
    # Create the Vectorizer (ignores common English stop words)
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on existing FAQs and transform them
    tfidf_matrix = vectorizer.fit_transform(faq_questions)
    
    # Transform the user's input query
    user_vector = vectorizer.transform([user_query])
    
    # Calculate cosine similarity between user query and all FAQ questions
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Find the index of the highest similarity score
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # Debug information to show how NLP processed the query
    debug_info = {
        "matched_question": faq_questions[best_match_idx],
        "similarity_score": round(best_score, 4),
        "all_scores": dict(zip(faq_questions, [round(s, 4) for s in similarities]))
    }
    
    # If the similarity is high enough, return the matching answer
    if best_score >= threshold:
        return faq_answers[best_match_idx], debug_info
    else:
        fallback_msg = "I'm sorry, I couldn't quite find an exact answer to that. Could you try rephrasing your question? Or, type 'contact agent' to reach our support team."
        return fallback_msg, debug_info

# ==========================================
# 3. INTERACTIVE CHAT INTERFACE
# ==========================================
st.title("💬 Simple Customer Service Chatbot")
st.write(
    "This system uses TF-IDF and Cosine Similarity to understand user intent "
    "and map queries to the correct responses without requiring rigid keyword matches."
)

# Sidebar with background information and controls
st.sidebar.header("🔧 Chatbot Configuration")
similarity_threshold = st.sidebar.slider(
    "Similarity Threshold (Confidence)", 
    min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Higher threshold means the bot requires a closer match to reply. Lower values might trigger false positives."
)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Sample FAQs to Try:")
for q in faq_questions:
    st.sidebar.markdown(f"- *{q}*")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your automated assistant. How can I help you today?"}
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input Box
if user_input := st.chat_input("Ask me about shipping, returns, business hours, etc..."):
    
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    # Generate response
    bot_reply, debug_log = get_bot_response(user_input, threshold=similarity_threshold)
    
    # Display bot's response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.write(bot_reply)
        
        # Optional: Show how the NLP matched the query under the hood
        with st.expander("🔍 See NLP Matching Details"):
            st.json({
                "User Input": user_input,
                "Best Match Found": debug_log["matched_question"],
                "Confidence Score": debug_log["similarity_score"],
                "Threshold Setup": similarity_threshold,
                "Status": "Success" if debug_log["similarity_score"] >= similarity_threshold else "Fell back to default response"
            })