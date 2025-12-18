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
from zoneinfo import ZoneInfo  # ✅ IMPORTANT

from flask import Flask, render_template, request, jsonify, send_from_directory

# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__, static_folder="static", template_folder="templates")

# =====================================================
# TIMEZONE CONFIG (CHANGE ONLY THIS IF NEEDED)
# =====================================================
LOCAL_TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")
# Examples:
# ZoneInfo("Asia/Kolkata")
# ZoneInfo("Asia/Singapore")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
INTENTS_PATH = BASE_DIR / "intents.json"
WORDS_PKL = BASE_DIR / "words.pkl"
CLASSES_PKL = BASE_DIR / "classes.pkl"
MODEL_FILE = BASE_DIR / "venubot_model.h5"
AVATAR_PATH = BASE_DIR / "static" / "Avtar.png"

# =====================================================
# VALIDATE FILES
# =====================================================
required_files = [INTENTS_PATH, WORDS_PKL, CLASSES_PKL, MODEL_FILE, AVATAR_PATH]
missing = [str(f) for f in required_files if not f.exists()]

if missing:
    raise FileNotFoundError(
        "❌ Missing required files:\n" + "\n".join(missing)
    )

# =====================================================
# NLTK + MODEL LOAD
# =====================================================
lemmatizer = WordNetLemmatizer()

try:
    nltk.word_tokenize("test")
except Exception:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

with open(INTENTS_PATH, encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open(WORDS_PKL, "rb"))
classes = pickle.load(open(CLASSES_PKL, "rb"))
model = load_model(MODEL_FILE)

# =====================================================
# USER NAME MEMORY
# =====================================================
user_name = None

def extract_user_name(message):
    msg = message.lower()
    triggers = ["i am ", "i'm ", "im ", "my name is "]
    for t in triggers:
        if t in msg:
            name = message[msg.index(t) + len(t):].split()[0]
            if name.isalpha():
                return name.capitalize()
    return None

def personalize_reply(reply):
    if not user_name:
        return reply
    if user_name.lower() in reply.lower():
        return reply
    return reply

# =====================================================
# NLP HELPERS
# =====================================================
def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    sentence_words = set(clean_up_sentence(sentence))
    return np.array([1 if w in sentence_words else 0 for w in words], dtype=np.float32)

def predict_class(sentence, threshold=0.55):
    bow = bag_of_words(sentence)
    preds = model.predict(np.array([bow]), verbose=0)[0]
    results = [(classes[i], float(p)) for i, p in enumerate(preds) if p > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I can't understand that yet."

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    now = datetime.now(LOCAL_TIMEZONE)
    hour = now.hour

    if 3 <= hour < 12:
        greeting = "Good morning"
    elif 12 <= hour < 18:
        greeting = "Good afternoon"
    elif 18 <= hour < 22:
        greeting = "Good evening"
    else:
        greeting = "Good night"

    return render_template(
        "index.html",
        avatar_url="/avatar",
        startup_greeting=greeting
    )

@app.route("/avatar")
def avatar():
    return send_from_directory(AVATAR_PATH.parent, AVATAR_PATH.name)

@app.route("/chat", methods=["POST"])
def chat():
    global user_name

    data = request.get_json()
    message = (data.get("message") or "").strip()

    detected_name = extract_user_name(message)
    if detected_name:
        user_name = detected_name
        return jsonify({
            "bot_message": f"Hi {user_name}! 🎶 Lovely to meet you!",
            "options": []
        })

    try:
        if not message:
            reply = "🎵 Say something and I'll sing back!"
        else:
            intents_pred = predict_class(message)
            if intents_pred:
                reply = get_response(intents_pred[0][0])
            else:
                reply = "Sorry, I can't understand your query."

        reply = personalize_reply(reply)

    except Exception:
        traceback.print_exc()
        reply = "⚠️ Something went wrong. Please try again."

    return jsonify({"bot_message": reply, "options": []})

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
