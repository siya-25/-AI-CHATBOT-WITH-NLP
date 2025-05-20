# ChatBuddy - A Simple NLP Chatbot

ChatBuddy is a Python-based chatbot that uses Natural Language Processing (NLP) to understand and respond to user queries. It uses NLTK for text preprocessing and scikit-learn for pattern matching.

## Features

- Natural language understanding using NLTK
- Pattern matching using TF-IDF and cosine similarity
- Dynamic responses based on user input
- Time-based responses
- Personalized responses using user's name

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the chatbot:
   ```bash
   python chatbot.py
   ```
2. Enter your name when prompted
3. Start chatting! The chatbot can:
   - Greet you
   - Tell you the current time
   - Answer questions about itself
   - Respond to thank you messages
   - Say goodbye

## Example Conversations

```
ChatBuddy: Hi there! What's your name?
You: John
ChatBuddy: Nice to meet you, John! How can I assist you today?
(Type 'exit' to end the conversation.)
You: what time is it?
ChatBuddy: The current time is 02:30 PM
You: what can you do?
ChatBuddy: I can help you with various things! Just ask me questions or chat with me.
You: bye
ChatBuddy: Goodbye John! Have a great day!
```

## Customization

You can customize the chatbot's responses by editing the `intents.json` file. Each intent contains:
- `tag`: The category of the intent
- `patterns`: Example phrases that match this intent
- `responses`: Possible responses for this intent

## Requirements

- Python 3.6+
- NLTK
- scikit-learn
- numpy

AI CHATBOT WITH NLP


