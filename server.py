import os
import nltk
import numpy as np
import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

print("--> Initializing NLP Environment...")
nltk.download('punkt', quiet=True)

# 1. Load the fine-tuned BERT model
model_dir = './bert_claim_model'
if os.path.exists(model_dir):
    print("--> Loading fine-tuned BERT claim model...")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
else:
    print(f"[ERROR] {model_dir} missing. Run fine_tune_bert.py first.")
    exit(1)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    raw_text = data['text'].strip()
    sentences = [s.strip() for s in nltk.tokenize.sent_tokenize(raw_text) if s.strip()]
    
    if not sentences:
        return jsonify({
            "document_chunks": [],
            "metrics": {
                "total_sentences": 0,
                "claim_count": 0,
                "confidence": "0%"
            }
        })
    
    document_chunks = []
    total_confidence = 0
    claim_count = 0
    
    for i, sentence in enumerate(sentences):
        prev_sent = sentences[i-1] if i > 0 else ""
        target_sent = sentences[i]
        next_sent = sentences[i+1] if i < len(sentences) - 1 else ""
        
        formatted_text = f"Prev: {prev_sent} | Target: {target_sent} | Next: {next_sent}"
        
        # Tokenize and predict
        inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]
            
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label]) * 100
        
        is_claim = (pred_label == 1)
        
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
        
        # Dynamic path to the local UI.html
        ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "UI.html"))
        webbrowser.open(f"file:///{ui_path.replace(os.sep, '/')}") 

    # Launch the browser automatically on a safe background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start your original Flask engine
    app.run(host="127.0.0.1", port=8000, debug=False)