import streamlit as st
import joblib

def preprocess_text(text):
    text = text.lower()
    return text

# Function to load pre-trained sentiment analysis models
def load_sentiment_models():
    models = []
    for i in range(1, 4):
        model = joblib.load(f'/Users/ronitkhurana/PycharmProjects/cogntive/Project/model_dumps/sentiment_model_{i}.joblib')
        # Load TF-IDF vectorizer
        tfidf_vectorizer = joblib.load(f'/Users/ronitkhurana/PycharmProjects/cogntive/Project/model_dumps/tfidf_vectorizer_{i}.joblib')
        models.append((model, tfidf_vectorizer))
    return models

# Function to perform sentiment analysis
def perform_sentiment_analysis(text, model_index):
    models = load_sentiment_models()
    model, tfidf_vectorizer = models[model_index - 1]
    # Preprocess text and transform using loaded TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([preprocess_text(text)])
    return sent(model.predict(text_tfidf)[0])

# Function to load pre-trained LDA model
def load_lda_model():
    lda_model = joblib.load('/Users/ronitkhurana/PycharmProjects/cogntive/Project/model_dumps/lda_model.joblib')
    return lda_model

# Function to perform topic modeling
def perform_topic_modeling(text):
    lda_model = load_lda_model()
    vectorizer = joblib.load('/Users/ronitkhurana/PycharmProjects/cogntive/Project/model_dumps/lda_vectorizer.joblib')
    text = preprocess_text(text)
    X = vectorizer.transform([text])
    return lda_model.transform(X)

def sent(x):
    if x == ' ':
        return 'Neutral'
    elif int(x) > 9:
        return 'Positive'
    elif int(x) >= 4:
        return 'Neutral'
    else:
        return 'Negative'

def main():
    st.title("Customer Service Chat Analysis")
    st.write("Welcome to the Customer Service Chat Analysis app!")
    st.write("For the best results, try using travel-related words such as 'flight', 'hotel', 'service', 'booking', etc.")
    st.sidebar.title("Model options")


    # Sidebar options
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Topic Modeling", "Sentiment Analysis"])

    if analysis_type == "Topic Modeling":
        st.header("Topic Modeling")

        # Get user input
        text_input = st.text_area("Enter your chat message:", "")

        # Perform topic modeling
        if st.button("Analyze Topic"):
            if text_input:
                topic_distribution = perform_topic_modeling(text_input)
                st.write("Topic Distribution:")
                st.write(topic_distribution)
                st.bar_chart(topic_distribution[0])



    elif analysis_type == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        st.info("Try changing the model from sidebar to see different results")

        # Get user input
        text_input = st.text_area("Enter your chat message:", "")
        model_index = st.sidebar.selectbox("Select Sentiment Analysis Model", [1, 2, 3])

        # Perform sentiment analysis
        if st.button("Analyze Sentiment"):
            if text_input:
                sentiment = perform_sentiment_analysis(text_input, model_index)
                st.write(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
