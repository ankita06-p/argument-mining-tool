import os
import nltk
import numpy as np
import tensorflow_hub as hub
from keras import layers
from sklearn.metrics import classification_report
import Preprocessing.araucaria_dataset
import tensorflow as tf
import keras.callbacks
import keras.metrics
import pickle
import sys

# ---------------------------------------------------------------------------
# 1.  Load data  (document-level train/test split done inside the loader)
# ---------------------------------------------------------------------------
X_sent_train, Y_train, X_sent_test, Y_test = \
    Preprocessing.araucaria_dataset.getDataLabelledSentences()

Y_train = np.array(Y_train)
Y_test  = np.array(Y_test)

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
cache_files = [
    'araucaria_embedded_train.pkl',
    'araucaria_embedded_test.pkl',
    'araucaria_Y_train.pkl',
    'araucaria_Y_test.pkl',
]
need_embed = not all(os.path.exists(f) for f in cache_files)

if not need_embed:
    print("Loading cached embeddings...")
    with open('araucaria_embedded_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('araucaria_embedded_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('araucaria_Y_train.pkl', 'rb') as f:
        Y_train = pickle.load(f)
    with open('araucaria_Y_test.pkl', 'rb') as f:
        Y_test = pickle.load(f)
else:
    batch_size = 1000

    print("Embedding training set in batches...")
    X_train = []
    for i in range(0, len(X_sent_train), batch_size):
        batch = X_sent_train[i:i+batch_size]
        emb = embed(batch)
        X_train.extend(emb.numpy())
    X_train = np.array(X_train)

    print("Embedding test set in batches...")
    X_test = []
    for i in range(0, len(X_sent_test), batch_size):
        batch = X_sent_test[i:i+batch_size]
        emb = embed(batch)
        X_test.extend(emb.numpy())
    X_test = np.array(X_test)

    # Persist to disk
    with open('araucaria_Y_train.pkl', 'wb') as f:
        pickle.dump(Y_train, f)
    with open('araucaria_Y_test.pkl', 'wb') as f:
        pickle.dump(Y_test, f)
    with open('araucaria_embedded_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('araucaria_embedded_test.pkl', 'wb') as f:
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
model_file = 'argument_model_v2.keras'
if os.path.exists(model_file):
    print("Loading saved model from disk...")
    model = keras.models.load_model(model_file)
    model.summary()
else:
    print("No saved model found. Training from scratch...")
    model = keras.Sequential()
    model.add(
      keras.layers.Dense(
        units=256,
        input_shape=(X_train.shape[1], ),
        activation='relu'
      )
    )
    model.add(
      keras.layers.Dropout(rate=0.3)
    )
    model.add(
      keras.layers.Dense(
        units=128,
        activation='relu'
      )
    )
    model.add(
      keras.layers.Dropout(rate=0.3)
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
score = model.evaluate(X_test, Y_test, verbose=2)
print(f"Test loss / metrics: {score}")

thresh = 0.5
y_pred = model.predict(X_test)
y_pred_labels = [1 if p > thresh else 0 for p in y_pred]

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT (AraucariaDB test set)")
print("=" * 60)
print(classification_report(Y_test, y_pred_labels, target_names=["Not claim", "Claim"]))

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

            sentences = nltk.tokenize.sent_tokenize(user_input)
            for sentence in sentences:
                emb = embed([sentence])
                sent_emb = tf.reshape(emb, [-1]).numpy()

                # Predict
                pred = model.predict(np.array([sent_emb]), verbose=0)[0][0]
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
