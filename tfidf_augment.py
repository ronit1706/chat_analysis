import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/customer_service_data.csv")

data.dropna(subset=['Text'], inplace=True)

data = data[data['Customer Rating'] != ' ']

data['Customer Rating'] = data['Customer Rating'].astype(int)

def augment_data(text, rating):

    ascii_values = np.frombuffer(text.encode(), dtype=np.uint8)
    noise = np.random.normal(loc=0, scale=1, size=len(ascii_values)).astype(np.uint8)
    augmented_ascii = ascii_values + noise
    augmented_ascii = np.clip(augmented_ascii, 0, 127)
    augmented_text = augmented_ascii.tobytes().decode(errors='replace')
    augmented_rating = min(max(int(np.random.normal(float(rating), 0.5)), int(rating) - 1), int(rating) + 1)
    return augmented_text, augmented_rating


augmented_texts = []
augmented_ratings = []
for _, row in data.iterrows():
    text = row['Text']
    rating = row['Customer Rating']
    for _ in range(5):
        augmented_text, augmented_rating = augment_data(text, rating)
        augmented_texts.append(augmented_text)
        augmented_ratings.append(augmented_rating)

augmented_data = pd.DataFrame({'Text': augmented_texts, 'Customer Rating': augmented_ratings})
data = pd.concat([data, augmented_data], ignore_index=True)

data.info()


X = data['Text']
y = data['Customer Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

joblib.dump(tfidf_vectorizer, 'model_dumps/tfidf_vectorizer_3.joblib')

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_train_tfidf, y_train)

y_pred = sentiment_model.predict(X_test_tfidf)

sentiment_accuracy = accuracy_score(y_test, y_pred)
print("Sentiment Analysis Accuracy:", sentiment_accuracy)

joblib.dump(sentiment_model, 'model_dumps/sentiment_model_3.joblib')
