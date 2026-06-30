import os
import nltk
import numpy as np
import tensorflow_hub as hub
from keras import layers
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import Preprocessing.araucaria_dataset
import Preprocessing.aaec_dataset
import tensorflow as tf
import keras.callbacks
import keras.metrics
import pickle
import sys

# ---------------------------------------------------------------------------
# 1.  Load data  (document-level train/test split done inside the loader)
# ---------------------------------------------------------------------------
araucaria_train_docs, araucaria_train_labels, araucaria_test_docs, araucaria_test_labels = \
    Preprocessing.araucaria_dataset.getDataLabelledDocuments()

aaec_train_docs, aaec_train_labels, aaec_test_docs, aaec_test_labels = \
    Preprocessing.aaec_dataset.getDataLabelledDocuments()

# Merge documents and labels
train_docs = araucaria_train_docs + aaec_train_docs
train_labels = araucaria_train_labels + aaec_train_labels
test_docs = araucaria_test_docs + aaec_test_docs
test_labels = araucaria_test_labels + aaec_test_labels

# ---------------------------------------------------------------------------
# 2.  Load Universal Sentence Encoder
# ---------------------------------------------------------------------------
try:
    embed = hub.load("universal-sentence-encoder_4")
except Exception:
    print("Local universal-sentence-encoder_4 not found. Downloading from TF Hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# ---------------------------------------------------------------------------
# 3.  Embed sentences  (with pickle caching)
# ---------------------------------------------------------------------------
threshold = Preprocessing.araucaria_dataset.SIMILARITY_THRESHOLD
train_emb_cache = f'merged_{threshold}_context_embedded_train.pkl'
test_emb_cache = f'merged_{threshold}_context_embedded_test.pkl'
train_y_cache = f'merged_{threshold}_context_Y_train.pkl'
test_y_cache = f'merged_{threshold}_context_Y_test.pkl'

cache_files = [train_emb_cache, test_emb_cache, train_y_cache, test_y_cache]
need_embed = not all(os.path.exists(f) for f in cache_files)

if not need_embed:
    print(f"Loading cached context embeddings for threshold {threshold}...")
    with open(train_emb_cache, 'rb') as f:
        X_train = pickle.load(f)
    with open(test_emb_cache, 'rb') as f:
        X_test = pickle.load(f)
    with open(train_y_cache, 'rb') as f:
        Y_train = pickle.load(f)
    with open(test_y_cache, 'rb') as f:
        Y_test = pickle.load(f)
else:
    print("Generating contextual embeddings for training documents...")
    X_train = []
    Y_train = []
    
    for doc_sents, doc_lbls in zip(train_docs, train_labels):
        if not doc_sents:
            continue
        doc_embs = embed(doc_sents).numpy()  # shape: [len(doc_sents), 512]
        
        for j in range(len(doc_sents)):
            prev_emb = doc_embs[j-1] if j > 0 else np.zeros(512)
            target_emb = doc_embs[j]
            next_emb = doc_embs[j+1] if j < len(doc_sents) - 1 else np.zeros(512)
            
            context_emb = np.concatenate([prev_emb, target_emb, next_emb])
            X_train.append(context_emb)
            Y_train.append(doc_lbls[j])
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print("Generating contextual embeddings for test documents...")
    X_test = []
    Y_test = []
    
    for doc_sents, doc_lbls in zip(test_docs, test_labels):
        if not doc_sents:
            continue
        doc_embs = embed(doc_sents).numpy()  # shape: [len(doc_sents), 512]
        
        for j in range(len(doc_sents)):
            prev_emb = doc_embs[j-1] if j > 0 else np.zeros(512)
            target_emb = doc_embs[j]
            next_emb = doc_embs[j+1] if j < len(doc_sents) - 1 else np.zeros(512)
            
            context_emb = np.concatenate([prev_emb, target_emb, next_emb])
            X_test.append(context_emb)
            Y_test.append(doc_lbls[j])
            
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Persist to disk
    with open(train_y_cache, 'wb') as f:
        pickle.dump(Y_train, f)
    with open(test_y_cache, 'wb') as f:
        pickle.dump(Y_test, f)
    with open(train_emb_cache, 'wb') as f:
        pickle.dump(X_train, f)
    with open(test_emb_cache, 'wb') as f:
        pickle.dump(X_test, f)

print(f"Train shape: {X_train.shape}  Labels: {Y_train.shape}")
print(f"Test  shape: {X_test.shape}  Labels: {Y_test.shape}")

# ---------------------------------------------------------------------------
# 4.  Compute dynamic class weights from actual label distribution
# ---------------------------------------------------------------------------
n_neg = int(np.sum(Y_train == 0))
n_pos = int(np.sum(Y_train == 1))
total = n_neg + n_pos
weight_for_0 = total / (2.0 * n_neg) if n_neg else 1.0
weight_for_1 = total / (2.0 * n_pos) if n_pos else 1.0
class_weight = {0: weight_for_0, 1: weight_for_1}
print(f"Class weights: {class_weight}")

# ---------------------------------------------------------------------------
# 5.  Build / load model
# ---------------------------------------------------------------------------
model_file = 'argument_model_v5_merged.keras'
if os.path.exists(model_file):
    print("Loading saved model from disk...")
    model = keras.models.load_model(model_file)
    model.summary()
else:
    print("No saved model found. Training from scratch...")
    model = keras.Sequential()
    model.add(
      keras.layers.Dense(
        units=64,
        input_shape=(X_train.shape[1], ),
        activation='relu'
      )
    )
    model.add(
      keras.layers.Dropout(rate=0.5)
    )
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )

    # Start training
    history = model.fit(
        X_train, Y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        shuffle=True,
        class_weight=class_weight,
        callbacks=[es]
    )

    print("Saving model to disk...")
    model.save(model_file)

# ---------------------------------------------------------------------------
# 6.  Evaluate on held-out test set
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5b. Train a Baseline Logistic Regression Classifier
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training Logistic Regression Baseline...")
print("=" * 60)
sample_weights = np.array([class_weight[y] for y in Y_train])
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, Y_train, sample_weight=sample_weights)
y_pred_lr = lr_model.predict(X_test)

# ---------------------------------------------------------------------------
# 6.  Evaluate on held-out test set
# ---------------------------------------------------------------------------
score = model.evaluate(X_test, Y_test, verbose=2)
print(f"Neural Network Test loss / metrics: {score}")

thresh = 0.5
y_pred_nn = model.predict(X_test)
y_pred_nn_labels = [1 if p > thresh else 0 for p in y_pred_nn]

print("\n" + "=" * 60)
print("EVALUATION & COMPARISON")
print("=" * 60)

print("\n" + "-" * 30 + " LOGISTIC REGRESSION BASELINE " + "-" * 30)
print(classification_report(Y_test, y_pred_lr, target_names=["Not claim", "Claim"]))

print("\n" + "-" * 30 + " SIMPLIFIED NEURAL NETWORK " + "-" * 30)
print(classification_report(Y_test, y_pred_nn_labels, target_names=["Not claim", "Claim"]))

# ---------------------------------------------------------------------------
# 7.  Interactive Sentence Prediction Loop
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("Argument Mining Interactive Mode")
print("Type a sentence to check if it's a claim, or type 'exit' to quit.")
print("=" * 50)

if not sys.stdin.isatty():
    print("\nNon-interactive environment detected. Skipping interactive mode.")
else:
    while True:
        try:
            user_input = input("\nEnter a sentence: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print("Exiting interactive mode. Goodbye!")
                break

            sentences = [s.strip() for s in nltk.tokenize.sent_tokenize(user_input) if s.strip()]
            if not sentences:
                continue
            
            doc_embs = embed(sentences).numpy()  # shape: [len(sentences), 512]
            
            for j, sentence in enumerate(sentences):
                prev_emb = doc_embs[j-1] if j > 0 else np.zeros(512)
                target_emb = doc_embs[j]
                next_emb = doc_embs[j+1] if j < len(sentences) - 1 else np.zeros(512)
                
                context_emb = np.concatenate([prev_emb, target_emb, next_emb])

                # Predict
                pred = model.predict(np.array([context_emb]), verbose=0)[0][0]
                is_claim = 1 if pred > thresh else 0

                print(f"\nSentence: \"{sentence}\"")
                if is_claim == 1:
                    print(f"Prediction: CLAIM (Confidence: {pred*100:.1f}%)")
                else:
                    print(f"Prediction: NOT A CLAIM (Confidence: {(1-pred)*100:.1f}%)")
        except KeyboardInterrupt:
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
