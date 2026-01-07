# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Dataset
texts = [
    "I love this product",
    "Very bad experience",
    "Amazing quality",
    "Not worth the money",
    "Excellent performance",
    "Terrible support",
    "Happy with purchase",
    "Waste of time",
    "Good value",
    "Disappointed"
]

sentiments = [
    "Positive", "Negative", "Positive", "Negative", "Positive",
    "Negative", "Positive", "Negative", "Positive", "Negative"
]

# Encode sentiment labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(sentiments)  # Positive=1, Negative=0

# Create a pipeline: Vectorizer + Naive Bayes
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(texts, y)

# Test with new samples
test_sentences = [
    "Excellent product",
    "Very disappointing experience"
]

predictions = model.predict(test_sentences)
predicted_labels = label_encoder.inverse_transform(predictions)

# Display results
for text, sentiment in zip(test_sentences, predicted_labels):
    print(f"Text: '{text}' → Sentiment: {sentiment}")
