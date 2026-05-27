import os
import nltk
import numpy as np
import tensorflow_hub as hub
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import Preprocessing.wikipedia_dataset
import tensorflow as tf
import keras.callbacks
import keras.metrics
import pickle
import random

# Get X (sentences) and Y from the original dataset
X, Y = Preprocessing.wikipedia_dataset.getDataLabelledSentences()
print("X length is {}".format(len(X)))
print("Y length is {}".format(len(Y)))

# Remove words from the sentences (eg stopwords, proper nouns...)
#for sentence in X:
#    tagged_sentence = nltk.tag.pos_tag(sentence.split())
#    print(tagged_sentence)
#    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NN' and tag != 'NNPS' and tag != 'NNP']

#    print(' '.join(edited_sentence))
#    X_updated.append(' '.join(edited_sentence))



Y = np.array(Y)
X_train = []
X_test = []

# Load the Universal Sentence Encoder model (needed for embedding generation and final predictions)
try:
    embed = hub.load("universal-sentence-encoder_4")
except Exception:
    print("Local universal-sentence-encoder_4 not found. Downloading from TF Hub...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# If false, save the embeddings as pickle to save time next time
required_files = ['sentences_embedded_train.pkl', 'sentences_embedded_test.pkl', 'Y_train.pkl', 'Y_test.pkl']
Reload_embeddings = not all(os.path.exists(f) for f in required_files)
if not Reload_embeddings:
    with open('sentences_embedded_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('sentences_embedded_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('Y_train.pkl', 'rb') as f:
        Y_train = pickle.load(f)
    with open('Y_test.pkl', 'rb') as f:
        Y_test = pickle.load(f)

else:
    # Divide data in train and test
    train_len = int(len(X) / 100 * 80)

    X_sent_train, X_sent_test, Y_train, Y_test = \
        train_test_split(
            X,
            Y,
            test_size=.1
        )

    # embed sentences in train and test using USE
    # You must download https://tfhub.dev/google/universal-sentence-encoder/4 and extract it
    print("Embedding training set in batches...")
    batch_size = 1000
    for i in range(0, len(X_sent_train), batch_size):
        batch = X_sent_train[i:i+batch_size]
        emb = embed(batch)
        X_train.extend(emb.numpy())
    X_train = np.array(X_train)

    print("Embedding test set in batches...")
    for i in range(0, len(X_sent_test), batch_size):
        batch = X_sent_test[i:i+batch_size]
        emb = embed(batch)
        X_test.extend(emb.numpy())
    X_test = np.array(X_test)

    # Save X and Y as pickle
    with open('Y_train.pkl', 'wb') as f:
        pickle.dump(Y_train, f)
    with open('Y_test.pkl', 'wb') as f:
        pickle.dump(Y_test, f)
    with open('sentences_embedded_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('sentences_embedded_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)


print(X_train.shape, Y_train.shape)

# Set the ratio of positive and negative example
X_train_diminued = []
Y_train_diminued = []
# TODO vectorize this
percentage_chance = 0
i = 0
for idx, sent in enumerate(X_train):
    if random.random() < percentage_chance and Y_train[idx] == 0:
        # Example at this idx is not kept
        continue
    else:
        X_train_diminued.append(sent)
        Y_train_diminued.append(Y_train[idx])
        i = i + 1
        # Example at this idx is kept

X_train = np.array(X_train_diminued)
Y_train = np.array(Y_train_diminued)

print(X_train.shape, Y_train.shape)

# Create neural network
model_file = 'argument_model.keras'
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
      keras.layers.Dropout(rate=0.1)
    )
    model.add(
      keras.layers.Dense(
        units=128,
        activation='relu'
      )
    )
    model.add(
      keras.layers.Dropout(rate=0.1)
    )
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])

    # validation set may be too small for this being really effective
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=2,
                                  verbose=0, mode='auto')

    # start training
    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        shuffle=True,
        class_weight= {0: 1.,
                    1: 10}
    )
    
    print("Saving model to disk...")
    model.save(model_file)

score=model.evaluate(X_test, Y_test, verbose=2)
print(score)

predictions = model.predict(X_test)


# Predict on the test data
y_pred = model.predict(X_test)

thresh = 0.5
y_pred = [1 if a_ > thresh else 0 for a_ in y_pred]

# Print resuts
print(classification_report(Y_test, y_pred, target_names=["Not claim", "Claim"]))

# Print the claims in included in text files of the following directory
test_dir_path = "Datasets/Wikipedia/test"
if not os.path.exists(test_dir_path):
    print(f"Creating missing directory: {test_dir_path}")
    os.makedirs(test_dir_path, exist_ok=True)
    src_dir = "Datasets/Wikipedia/articles"
    if os.path.exists(src_dir):
        import shutil
        sample_files = [f for f in os.listdir(src_dir) if f.endswith(".txt")][:3]
        for f in sample_files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(test_dir_path, f))
        print(f"Copied sample files {sample_files} to {test_dir_path} for testing predictions.")

directory = os.fsencode(test_dir_path)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.endswith(".txt") or filename.endswith(".py"):
        with open(os.path.join(directory, file), 'r', encoding='utf-8') as txt_file:
            txt = txt_file.read().replace('\n', '')
            # print(txt)
            sentences = nltk.tokenize.sent_tokenize(txt)
            sentences_emb = []
            for sentence in sentences:
                # print(r)
                emb = embed([sentence])
                sent_emb = tf.reshape(emb, [-1]).numpy()
                sentences_emb.append(sent_emb)

            sentences_emb = np.array(sentences_emb)

            predictions = model.predict(sentences_emb)
            predictions = [1 if a_ > thresh else 0 for a_ in predictions]

            for idx, prediction in enumerate(predictions):
                if prediction == 0:
                    continue
                    print("not claim")
                else:
                    print(sentences[idx])
                    print("A claim")
                #print(prediction)

# --- Interactive Sentence Prediction Loop ---
print("\n" + "="*50)
print("Argument Mining Interactive Mode")
print("Type a sentence to check if it's a claim, or type 'exit' to quit.")
print("="*50)

import sys
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



