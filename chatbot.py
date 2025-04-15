import json
import random
import datetime
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import os
import sys
import re
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        sys.exit(1)

def load_intents():
    try:
        with open("intents.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: intents.json file not found!")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: intents.json file is not valid JSON!")
        sys.exit(1)

def preprocess_text(text):
    try:
        # Tokenize and lemmatize
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

def analyze_sentiment(text):
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0

def solve_math_expression(expression):
    try:
        # Remove any non-math characters
        expression = re.sub(r'[^0-9+\-*/(). ]', '', expression)
        # Evaluate the expression safely
        result = eval(expression)
        return f"The answer is {result}"
    except Exception as e:
        return "I couldn't solve that math problem. Please try a different one."

def create_model(input_dim, output_dim):
    try:
        model = Sequential([
            Dense(128, input_shape=(input_dim,), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(output_dim, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)

def get_response(user_input, model, vectorizer, label_encoder, responses, user_name):
    try:
        # Check for math expressions
        if any(op in user_input for op in ['+', '-', '*', '/']):
            return solve_math_expression(user_input)

        # Analyze sentiment
        sentiment = analyze_sentiment(user_input)
        
        # Preprocess user input
        processed_input = preprocess_text(user_input)
        if not processed_input:
            return "I'm having trouble understanding that. Could you try again?"

        input_vector = vectorizer.transform([processed_input]).toarray()
        
        # Get prediction probabilities
        prediction_probs = model.predict(input_vector, verbose=0)[0]
        best_match_idx = np.argmax(prediction_probs)
        best_match_score = prediction_probs[best_match_idx]
        
        # If confidence score is too low, return default response
        if best_match_score < 0.5:
            if sentiment < -0.3:
                return "I sense you might be feeling down. Is there anything I can do to help?"
            elif sentiment > 0.3:
                return "I'm glad you're feeling positive! How can I assist you today?"
            return "I'm not sure I understand. Could you rephrase that?"
        
        # Get the best matching tag
        best_tag = label_encoder.inverse_transform([best_match_idx])[0]
        
        # Get a random response for the matched tag
        response = random.choice(responses[best_tag])
        
        # Replace placeholders
        if "{time}" in response:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            response = response.replace("{time}", current_time)
        if "{date}" in response:
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            response = response.replace("{date}", current_date)
        if "{name}" in response:
            response = response.replace("{name}", user_name)
        
        return response
    except Exception as e:
        print(f"Error getting response: {e}")
        return "I'm having trouble processing that. Could you try again?"

def main():
    try:
        # Download required NLTK data
        download_nltk_data()

        # Initialize lemmatizer and stopwords
        global lemmatizer, stop_words
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        # Load intents
        intents = load_intents()

        # Prepare training data
        patterns = []
        tags = []
        responses = {}

        for intent in intents["intents"]:
            tag = intent["tag"]
            responses[tag] = intent["responses"]
            for pattern in intent["patterns"]:
                processed_pattern = preprocess_text(pattern)
                if processed_pattern:  # Only add if preprocessing was successful
                    patterns.append(processed_pattern)
                    tags.append(tag)

        if not patterns:
            print("Error: No valid patterns found in intents.json!")
            sys.exit(1)

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(patterns).toarray()

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(tags)

        # Create and train model
        model = create_model(X.shape[1], len(np.unique(y)))
        model.fit(X, y, epochs=100, batch_size=8, verbose=0)

        # Get user name
        user_name = input("ChatBuddy: Hi there! What's your name?\nYou: ").strip()
        if not user_name:
            user_name = "friend"

        print(f"ChatBuddy: Nice to meet you, {user_name}! How can I assist you today?\n(Type 'exit' to end the conversation.)")

        # Main conversation loop
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print(f"ChatBuddy: Goodbye {user_name}! Have a great day!")
                    break
                    
                response = get_response(user_input, model, vectorizer, label_encoder, responses, user_name)
                print("ChatBuddy:", response)
            except KeyboardInterrupt:
                print("\nChatBuddy: Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"ChatBuddy: I encountered an error. Let's try again.")
                continue

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

