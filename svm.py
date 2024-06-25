import pandas as pd
import matplotlib.pyplot as plt
import joblib
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

data = pd.read_csv("data/customer_service_data.csv")

data.dropna(subset=['Text'], inplace=True)
data.dropna(subset=['Customer Rating'], inplace=True)

non_null = data.drop(data[data['Customer Rating'] == ' '].index, inplace=False)

data.drop(data[data['Customer Rating'] == ' '].index, inplace=True)

def map_rating(rating):
    if rating == ' ':
        return 'Unknown'
    elif int(rating) >= 9:
        return 'Positive'
    elif int(rating) >= 4:
        return 'Neutral'
    else:
        return 'Negative'



data['Rating Category'] = data['Customer Rating'].apply(map_rating)


plt.figure(figsize=(8, 6))
data['Rating Category'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Rating Categories')
plt.xlabel('Rating Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

X = data['Text']
y_sentiment = data['Customer Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

joblib.dump(tfidf_vectorizer, 'model_dumps/tfidf_vectorizer_1.joblib')


sentiment_model = SVC()
sentiment_model.fit(X_train_tfidf, y_train)

y_pred = sentiment_model.predict(X_test_tfidf)

sentiment_accuracy = accuracy_score(y_test, y_pred)
print("Sentiment Analysis Accuracy:", sentiment_accuracy)

joblib.dump(sentiment_model, 'model_dumps/sentiment_model_4.joblib')


wordcloud = WordCloud(width=800, height=400).generate(' '.join(data['Text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Frequent Words')
plt.show()
