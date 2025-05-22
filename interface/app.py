import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
from models.llm import LocalLLM

app = Flask(__name__)

# Initialize the local LLM model (example model_name and path)
llm = LocalLLM(model_name="dummy-model", model_path="models/dummy-model-path")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # Use the LLM to generate a response
    response_text = llm.generate(user_input)
    return jsonify({'reply': response_text})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
