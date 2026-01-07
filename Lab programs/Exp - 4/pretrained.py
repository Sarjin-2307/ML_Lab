from transformers import pipeline

# Load the default sentiment analysis model
# This downloads a pre-trained model (e.g., distilbert-base-uncased-finetuned-sst-2-english)
# and its associated tokenizer and configuration
sentiment_analyzer = pipeline("sentiment-analysis")

# Perform sentiment analysis on a single text
text = "I love using Hugging Face models!"
result = sentiment_analyzer(text)
print(result)

# Perform sentiment analysis on multiple texts
texts = ["This service is terrible.", "The package arrived on time."]
results = sentiment_analyzer(texts)
print(results)
