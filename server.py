import os
import nltk
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras

app = Flask(__name__)
CORS(app)

print("--> Initializing NLP Environment...")
nltk.download('punkt', quiet=True)

# 1. Load Universal Sentence Encoder
try:
    embed = hub.load("universal-sentence-encoder_4")
except Exception:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# 2. Load the trained model
model_file = 'argument_model_v2.keras'
if os.path.exists(model_file):
    model = keras.models.load_model(model_file)
else:
    print(f"[ERROR] {model_file} missing. Run classification.py first.")
    exit(1)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    raw_text = data['text'].strip()
    sentences = nltk.tokenize.sent_tokenize(raw_text)
    
    document_chunks = []
    total_confidence = 0
    claim_count = 0
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
            
        # Get embeddings and predict
        emb = embed([sentence])
        sent_emb = tf.reshape(emb, [-1]).numpy()
        pred = float(model.predict(np.array([sent_emb]), verbose=0)[0][0])
        
        is_claim = pred > 0.5
        confidence = pred * 100 if is_claim else (1 - pred) * 100
        
        total_confidence += confidence
        if is_claim:
            claim_count += 1
            
        document_chunks.append({
            "id": f"s{i+1}",
            "type": "Claim" if is_claim else "Not Claim",
            "text": sentence,
            "confidence": round(confidence, 1)
        })
        
    avg_confidence = round(total_confidence / len(sentences), 1) if sentences else 0
    
    return jsonify({
        "document_chunks": document_chunks,
        "metrics": {
            "total_sentences": len(sentences),
            "claim_count": claim_count,
            "confidence": f"{avg_confidence}%"
        }
    })

if __name__ == "__main__":
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(2.0)  # Wait for the ML environment to initialize
        
        # This opens the file directly as a local system file. 
        # Your browser loads it instantly, and the HTML code will still talk to http://127.0.0.1:8000/api/analyze perfectly!
        webbrowser.open("file:///C:/Users/dharu/OneDrive/Desktop/argugraph_project/simple.html") 

    # Launch the browser automatically on a safe background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start your original Flask engine
    app.run(host="127.0.0.1", port=8000, debug=False)