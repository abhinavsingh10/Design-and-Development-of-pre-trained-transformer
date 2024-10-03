from flask import Flask, request, jsonify, Response
from flask_cors import CORS  
import tensorflow as tf
import json
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "http://127.0.0.1:5500"}})

# Load the model
model_path = "C:\\Users\\hp\\Downloads\\model (6).keras"
gpt = tf.keras.models.load_model(model_path)

# Load the vocabulary
vocab_path = "C:\\Users\\hp\\Downloads\\vocab (6).json"
with open(vocab_path, 'r') as vocab_file:
    vocab = json.load(vocab_file)

class TextGenerator():
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
        sample_token = None
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y, _ = gpt.predict(x, verbose=0)
            sample_token, _ = self.sample_from(y[0][-1], temperature)
            start_tokens.append(sample_token)
            yield self.index_to_word[sample_token]
        yield '' 

text_generator = TextGenerator(vocab)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    start_prompt = data.get('start_prompt', '')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.5)

    if not start_prompt:
        return jsonify({"error": "start_prompt is required"}), 400

    def generate_stream():
        yield start_prompt+' '
        for token in text_generator.generate(start_prompt, max_tokens, temperature):
            yield f"{token} "

    return Response(generate_stream(), content_type='text/plain;charset=utf-8')

if __name__ == '__main__':
    app.run(debug=True)
