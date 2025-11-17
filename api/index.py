import os
import json
import random
import pickle
import numpy as np
import nltk
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

nltk.download('punkt')

# === Path setup for Vercel ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

model = load_model(os.path.join(ROOT_DIR, "venubot_model.h5"))
words = pickle.load(open(os.path.join(ROOT_DIR, "words.pkl"), "rb"))
classes = pickle.load(open(os.path.join(ROOT_DIR, "classes.pkl"), "rb"))

with open(os.path.join(ROOT_DIR, "intents.json")) as file:
    intents = json.load(file)

app = Flask(__name__, static_folder=os.path.join(ROOT_DIR, "static"), template_folder=os.path.join(ROOT_DIR, "templates"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [word.lower() for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        for intent in intents_json["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "I didn't understand that. Can you rephrase?"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    intents_list = predict_class(msg)
    response = get_response(intents_list, intents)
    return jsonify({"response": response})

# Required for Vercel
def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == "__main__":
    app.run(debug=True)
