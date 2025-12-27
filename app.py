from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['Category', 'Message']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tf, y_train)

# Flask app
app = Flask(__name__)

# Extended list of spam keywords (commonly found in spam emails/SMS)
spam_words = [
    "free", "win", "winner", "cash", "prize", "urgent", "limited", "offer",
    "buy now", "click", "guarantee", "cheap", "bonus", "congratulations",
    "exclusive", "act now", "call now", "apply now", "100% free", "miracle",
    "trial", "deal", "credit", "loan", "investment", "pre-approved", "luxury",
    "money", "double your", "income", "earn", "weight loss", "no cost",
    "don't delete", "this isn't spam", "unsubscribe", "risk-free"
    
    
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "").lower().strip()

    if not message:
        return jsonify({"prediction": "invalid", "error": "Message is empty"})

    # Match full words only (case-insensitive)
    matched_words = []
    for word in spam_words:
        # Remove non-alphanumeric characters from spam keyword (for regex safety)
        word_pattern = r'\b' + re.escape(word.lower()) + r'\b'
        if re.search(word_pattern, message):
            matched_words.append(word)

    spam_word_count = len(matched_words)

    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    confidence = max(model.predict_proba(vec)[0])

    # Override if spammy words are strong indicator
    if pred == 0 and spam_word_count >= 2:
        pred = 1
        confidence = 0.99

    return jsonify({
        "prediction": int(pred),
        "confidence": round(confidence, 4),
        "spam_word_count": spam_word_count,
        "matched_words": matched_words
    })

if __name__ == "__main__":
    app.run(debug=True)

