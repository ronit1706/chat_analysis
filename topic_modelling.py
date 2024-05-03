import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load data
data = pd.read_csv("data/customer_service_data.csv")

# Drop rows with missing text
data.dropna(subset=['Text'], inplace=True)

# Vectorize text data
vectorizer = CountVectorizer(max_features=1000,
                             stop_words='english',
                             max_df=0.95,
                             min_df=2)
X = vectorizer.fit_transform(data['Text'])

joblib.dump(vectorizer, 'model_dumps/lda_vectorizer.joblib')

# Define number of topics
num_topics = 5

# Apply LDA
lda_model = LatentDirichletAllocation(n_components=num_topics,
                                      max_iter=10,
                                      learning_method='online',
                                      random_state=42)
lda_output = lda_model.fit_transform(X)

# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Display the top words for each topic
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))
        print()

# Number of words per topic
num_top_words = 10

# Display topics
print("Topics found via LDA:")
display_topics(lda_model, feature_names, num_top_words)

# Get the topic distribution for each document
doc_topic_dist = lda_model.transform(X)

# Calculate the proportion of documents assigned to each topic
topic_proportion = np.sum(doc_topic_dist, axis=0) / np.sum(doc_topic_dist)

# Plot the distribution of topics
plt.figure(figsize=(10, 6))
plt.bar(range(len(topic_proportion)), topic_proportion, color='skyblue')
plt.xlabel('Topic')
plt.ylabel('Proportion of Documents')
plt.title('Distribution of Topics')
plt.xticks(range(len(topic_proportion)), [f'Topic {i+1}' for i in range(len(topic_proportion))])
plt.show()

joblib.dump(lda_model, 'model_dumps/lda_model.joblib')