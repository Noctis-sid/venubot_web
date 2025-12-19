import traceback
import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# ---------------- App setup ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------- Paths ----------------
BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "intents.json"
WORDS_PKL = BASE_DIR / "words.pkl"
CLASSES_PKL = BASE_DIR / "classes.pkl"
MODEL_FILE = BASE_DIR / "venubot_model.h5"
AVATAR_PATH = BASE_DIR / "static" / "Avtar.png"

# ---------------- Validate files ----------------
for file in [INTENTS_PATH, WORDS_PKL, CLASSES_PKL, MODEL_FILE, AVATAR_PATH]:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")

# ---------------- NLTK ----------------
lemmatizer = WordNetLemmatizer()
try:
    nltk.word_tokenize("test")
except:
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# ---------------- Load intents ----------------
with open(INTENTS_PATH, encoding="utf-8") as f:
    data = json.load(f)

USER_RESPONSES = data.get("user_responses", [])
QUESTION_BANK = data.get("question_bank", [])
ALL_INTENTS = USER_RESPONSES + QUESTION_BANK

# ---------------- Load model ----------------
words = pickle.load(open(WORDS_PKL, "rb"))
classes = pickle.load(open(CLASSES_PKL, "rb"))
model = load_model(MODEL_FILE)

# ---------------- Conversation state ----------------
user_name = None
awaiting_name = False

# ---------------- Name helpers ----------------
def extract_user_name(message):
    msg = message.lower()
    triggers = ["i am ", "i'm ", "im ", "my name is "]
    for t in triggers:
        if t in msg:
            name_part = message[msg.index(t) + len(t):].strip()
            name = name_part.split()[0]
            if name.isalpha():
                return name.capitalize()
    return None

def personalize_reply(reply):
    if not user_name:
        return reply
    if user_name.lower() in reply.lower():
        return reply
    return reply

# ---------------- NLP helpers ----------------
def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words], dtype=np.float32)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [(classes[i], r) for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(tag):
    for intent in ALL_INTENTS:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I can't understand your query. You can ask anything from the suggestions below."

# ---------------- Options logic ----------------
PRIORITY_TAG = "content_yes"

def build_options():
    options = []

    # Always include video content option first
    for i in USER_RESPONSES:
        if i["tag"] == PRIORITY_TAG:
            options.append(i)
            break

    remaining = QUESTION_BANK.copy()
    random.shuffle(remaining)
    options.extend(remaining[:3])

    return options[:4]

def option_label(intent):
    if intent["tag"] == PRIORITY_TAG:
        return "Show me related videos"
    return intent["patterns"][0].capitalize()

# ---------------- Routes ----------------
@app.route("/")
def home():
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    return render_template("index.html", avatar_url="/avatar", startup_greeting=greet)

@app.route("/avatar")
def avatar():
    return send_from_directory(AVATAR_PATH.parent, AVATAR_PATH.name)

@app.route("/chat", methods=["POST"])
def chat():
    global user_name, awaiting_name

    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    option_tag = data.get("option_tag")

    # -------- Case 1: Waiting for name (single word like "sid") --------
    if awaiting_name and message and message.isalpha():
        user_name = message.capitalize()
        awaiting_name = False
        bot_text = f"Nice to meet you, {user_name}! 🎶 Do you love the sound of the flute?"
        return jsonify({
            "bot_message": bot_text,
            "options": [
                {"id": str(i+1), "label": option_label(opt), "tag": opt["tag"]}
                for i, opt in enumerate(build_options())
            ]
        })

    # -------- Case 2: Name in sentence ("i am sid") --------
    detected_name = extract_user_name(message)
    if detected_name:
        user_name = detected_name
        awaiting_name = False
        bot_text = f"Nice to meet you, {user_name}! 🎶 Do you love the sound of the flute?"
        return jsonify({
            "bot_message": bot_text,
            "options": [
                {"id": str(i+1), "label": option_label(opt), "tag": opt["tag"]}
                for i, opt in enumerate(build_options())
            ]
        })

    # -------- Case 3: Option clicked --------
    if option_tag:
        bot_text = get_response(option_tag)

    # -------- Case 4: Empty input --------
    elif not message:
        awaiting_name = True
        bot_text = "🎵 Say something and I’ll sing back! What’s your name?"

    # -------- Case 5: Normal message --------
    else:
        predictions = predict_class(message)
        if predictions and predictions[0][1] >= 0.55:
            bot_text = get_response(predictions[0][0])
        else:
            bot_text = "Sorry, I can't understand your query. You can ask anything from the suggestions below."

    bot_text = personalize_reply(bot_text)

    return jsonify({
        "bot_message": bot_text,
        "options": [
            {"id": str(i+1), "label": option_label(opt), "tag": opt["tag"]}
            for i, opt in enumerate(build_options())
        ]
    })

# ---------------- Run ----------------
if __name__ == "__main__":
    print("✅ VenuBot running successfully.")
    app.run(host="127.0.0.1", port=5000, debug=True)

