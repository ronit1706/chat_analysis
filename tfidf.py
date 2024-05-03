import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/customer_service_data.csv")

data.dropna(subset=['Text'], inplace=True)

def preprocess_text(text):
    text = text.lower()
    return text

data['Text'] = data['Text'].apply(preprocess_text)

X = data['Text']
y_sentiment = data['Customer Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

joblib.dump(tfidf_vectorizer, 'model_dumps/tfidf_vectorizer_2.joblib')

sentiment_model = LogisticRegression()
sentiment_model.fit(X_train_tfidf, y_train)

y_pred = sentiment_model.predict(X_test_tfidf)

sentiment_accuracy = accuracy_score(y_test, y_pred)
print("Sentiment Analysis Accuracy:", sentiment_accuracy)

joblib.dump(sentiment_model, 'model_dumps/sentiment_model_2.joblib')