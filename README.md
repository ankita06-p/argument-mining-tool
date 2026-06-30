# AM Workspace // Argument Mining Platform

AM Workspace is a computational argument mining platform designed to segment documents and automatically detect **Argumentative Claims** vs. **Contextual Non-Claims** in natural language sentences. 

The platform features a modern Tailwind CSS web interface, a lightweight Flask inference API server, and a state-of-the-art fine-tuned Transformer model.

---

## 🚀 Key Features

*   **Fine-Tuned Transformer Model**: Utilizes a customized sequence classification BERT model (`google/bert_uncased_L-2_H-128_A-2`) fine-tuned for high-recall claim detection.
*   **Context-Aware NLP Pipeline**: Evaluates sentences in context by structuring inputs as a sliding window of target sentences with their previous and next neighbors.
*   **Modular Dataset Merging**: Integrates both legal/political arguments (**AraucariaDB**) and persuasive student essays (**AAEC v2.0**) into a combined training corpus.
*   **Leakage-Free Splits**: Preprocessing separates training and test sets strictly at the *document level* to prevent cross-contamination.
*   **Zero-Overhead Inference**: The Flask server utilizes PyTorch on CPU, running without heavy TensorFlow initialization lag and reducing memory consumption by over 80%.

---

## 📊 Model Comparison & Metrics

All models were trained on 8,230 sentences and evaluated on 2,095 held-out test sentences:

| Model Architecture | Claim Recall | Claim F1-Score | Overall Accuracy | Inference Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (1536-D USE)** | 79% | 0.83 | 76% | ~10ms |
| **Keras Neural Network (1536-D USE)** | 82% | 0.84 | 77% | ~20ms |
| **Fine-Tuned BERT-Tiny (v6)** | **87%** | **0.84** | **77%** | ~50ms |

### Key Insight
Fine-tuning the internal attention mechanisms of the Transformer encoder allows **BERT-Tiny to reach the highest claim recall (87%)** across all models. It successfully captures local grammatical structures and indicators of argument boundaries.

---

## 📁 Repository Structure

```
ArgumentMining/
│
├── Preprocessing/                      # Dataset parser modules
│   ├── __init__.py
│   ├── araucaria_dataset.py            # Araucaria AIF parser (fuzzy matcher ratio >= 0.78)
│   ├── aaec_dataset.py                 # Student essays brat annotator & split parser
│   └── wikipedia_dataset.py            # Legacy wikipedia dataset loader
│
├── bert_claim_model/                   # Local serialized fine-tuned BERT model weights
│
├── UI.html                             # Tailwind CSS responsive canvas web interface
├── server.py                           # Flask REST API server (Torch & BERT prediction engine)
├── fine_tune_bert.py                   # PyTorch/Hugging Face Transformer fine-tuning script
├── classification.py                   # USE embedding extraction & Keras NN training script
│
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Excludes venv, cached pickles, and large models
└── README.md                           # Project walkthrough and documentation
```

---

## 🛠️ Installation & Getting Started

### 1. Setup Environment
Ensure Python 3.10+ is installed. Create a virtual environment and install the required dependencies:

```bash
# Initialize virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements
python -m pip install -r requirements.txt
```

### 2. Launch the Web Interface & Server
Start the Flask API server. This will automatically open the responsive **AM Workspace** interface in your default web browser:

```bash
python server.py
```
*   Pasting a text paragraph and clicking **Analyze Text** will query the fine-tuned BERT model and highlight Claims vs. Non-Claims along with prediction confidence levels.

### 3. Run Training / Fine-Tuning Pipeline
If you wish to retrain the Transformer model from scratch on CPU using the merged Araucaria + AAEC dataset (takes ~3 minutes with dynamic padding):

```bash
python fine_tune_bert.py
```
The script will tokenizer the corpus, train for 4 epochs, output the classification reports, and save the serialized weights to `./bert_claim_model/`.

---

## 📖 Citations & References

1.  **AraucariaDB**: Reed, C., & Rowe, G. (2004). Araucaria: Software for argument analysis, diagramming and representation. *International Journal on Artificial Intelligence Tools*, 13(04), 961-979.
2.  **AAEC v2.0**: Stab, C., & Gurevych, I. (2017). Parsing argumentation structures in persuasive essays. *Computational Linguistics*, 43(3), 619-659.
3.  **Universal Sentence Encoder**: Cer, D., et al. (2018). Universal sentence encoder. *arXiv preprint arXiv:1803.11175*.
