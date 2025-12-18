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
from zoneinfo import ZoneInfo   # ✅ IMPORTANT
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")

# ================== TIMEZONE CONFIG ==================
# Change ONLY this if needed
LOCAL_TIMEZONE = ZoneInfo("Asia/Kuala_Lumpur")
# Examples:
# Asia/Kolkata
# Asia/Singapore
# Asia/Jakarta
# =====================================================

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
        raise FileNotFoundError(f"Missing required file: {file}")

# ---------------- NLTK + model load ----------------
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

# ---------------- USER NAME MEMORY ----------------
user_name = None

def extract_user_name(message):
    msg = message.lower()
    triggers = ["i am ", "i'm ", "im ", "my name is "]
    for t in triggers:
        if t in msg:
            name = message[msg.index(t) + len(t):].strip().split()[0]
            if name.isalpha():
                return name.capitalize()
    return None

def personalize_reply(reply):
    if not user_name:
        return reply
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "good night"]
    low = reply.lower()
    for g in greetings:
        if low.startswith(g):
            return reply.replace(g, f"{g.capitalize()} {user_name}", 1)
    return reply

# ---------------- NLP helpers ----------------
def clean_up_sentence(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words], dtype=np.float32)

def predict_class(sentence, threshold=0.25):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": float(r[1])} for r in results]

def get_response_from_intent_tag(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I can't understand that."

def local_fallback():
    return "Sorry, I can't understand your query. You can ask anything from the suggestions below."

# ---------------- OPTIONS ----------------
PRIORITY_OPTION_TAG = "content_yes"

def build_options():
    all_intents = intents["intents"]
    priority = [i for i in all_intents if i["tag"] == PRIORITY_OPTION_TAG]
    remaining = [i for i in all_intents if i["tag"] != PRIORITY_OPTION_TAG]
    random.shuffle(remaining)
    return (priority + remaining)[:4]

def option_label(intent):
    if intent["tag"] == PRIORITY_OPTION_TAG:
        return "Show me related videos"
    return intent["patterns"][0].capitalize() if intent["patterns"] else intent["tag"]

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    now = datetime.now(LOCAL_TIMEZONE)
    hour = now.hour

    if 3 <= hour < 12:
        greet = "Good morning"
    elif 12 <= hour < 18:
        greet = "Good afternoon"
    elif 18 <= hour < 21:
        greet = "Good evening"
    else:
        greet = "Good night"

    return render_template("index.html",
                           avatar_url="/avatar",
                           startup_greeting=greet)

@app.route("/avatar")
def avatar():
    return send_from_directory(AVATAR_PATH.parent, AVATAR_PATH.name)

@app.route("/chat", methods=["POST"])
def chat():
    global user_name
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    option_tag = data.get("option_tag")

    detected_name = extract_user_name(message)
    if detected_name:
        user_name = detected_name
        reply = f"Hi {user_name}! Lovely to meet you 🎶"
    elif option_tag:
        reply = get_response_from_intent_tag(option_tag)
    elif message:
        preds = predict_class(message)
        if preds and preds[0]["probability"] >= 0.55:
            reply = get_response_from_intent_tag(preds[0]["intent"])
        else:
            reply = local_fallback()
    else:
        reply = "🎵 Say something and I'll sing back!"

    reply = personalize_reply(reply)

    options = build_options()
    options_json = [
        {"id": str(i + 1), "label": option_label(o), "tag": o["tag"]}
        for i, o in enumerate(options)
    ]

    return jsonify({"bot_message": reply, "options": options_json})

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("✅ VenuBot running with correct timezone")
    app.run(host="0.0.0.0", port=5000)
