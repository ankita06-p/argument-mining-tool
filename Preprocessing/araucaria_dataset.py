"""
Araucaria Dataset Preprocessing Module
=======================================
Parses AraucariaDB (AIF-format JSON + raw text) into labelled sentences
for binary claim detection. Returns document-level train/test splits
to prevent data leakage.

Dataset structure (per document):
  nodeset{N}.json  – AIF argument graph with typed nodes
  nodeset{N}.txt   – raw source text

Node types in AIF:
  "I"  = Information node (argumentative proposition: claim or premise)
  "RA" = Reasoning Application (support relation)
  "CA" = Conflict Application (attack relation)

We label each sentence from the raw text as:
  1 (Claim)     – if it substantially overlaps with any "I"-type node
  0 (Not Claim)  – otherwise
"""

import os
import json
import glob
import random
import nltk
from difflib import SequenceMatcher

# Ensure NLTK tokenizer data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Path to the araucaria folder (relative to the working directory, i.e. ArgumentMining/)
ARAUCARIA_DIR = os.path.join("Datasets", "Scientific articles", "araucaria")

# Minimum similarity ratio (0-1) for a sentence to be considered a match
# to an "I"-node.  0.7 handles minor whitespace / punctuation mismatches
# between the .json node text and the .txt source.
SIMILARITY_THRESHOLD = 0.6


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase for robust matching."""
    return " ".join(text.lower().split())


def _sentence_matches_any_node(sentence: str, node_texts: list[str]) -> bool:
    """Return True if *sentence* is a substantial match to any node text."""
    s_norm = _normalise(sentence)
    if not s_norm:
        return False
    for node_text in node_texts:
        n_norm = _normalise(node_text)
        if not n_norm:
            continue
        # Quick exact-substring check first (fast path)
        if n_norm in s_norm or s_norm in n_norm:
            return True
        # Fall back to fuzzy ratio
        ratio = SequenceMatcher(None, s_norm, n_norm).ratio()
        if ratio >= SIMILARITY_THRESHOLD:
            return True
    return False


def _parse_single_document(json_path: str, txt_path: str):
    """
    Parse one (json, txt) pair and return (sentences, labels).

    Returns
    -------
    sentences : list[str]
    labels    : list[int]   (1 = claim, 0 = not claim)
    """
    # --- Load the argument graph ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract all Information-node texts
    i_node_texts = [
        node["text"]
        for node in data.get("nodes", [])
        if node.get("type") == "I" and node.get("text", "").strip()
    ]

    if not i_node_texts:
        # No argumentative nodes in this document – skip entirely
        return [], []

    # --- Load and sentence-tokenise the raw text ---
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read().replace("\n", " ").strip()

    if not raw_text:
        return [], []

    sentences = nltk.tokenize.sent_tokenize(raw_text)

    # --- Label each sentence ---
    labels = []
    for sent in sentences:
        if _sentence_matches_any_node(sent, i_node_texts):
            labels.append(1)
        else:
            labels.append(0)

    return sentences, labels


def getDataLabelledSentences(
    data_dir: str = ARAUCARIA_DIR,
    test_ratio: float = 0.20,
    seed: int = 42,
):
    """
    Parse all Araucaria documents and return document-level train/test splits.

    Parameters
    ----------
    data_dir   : path to the araucaria folder
    test_ratio : fraction of *documents* held out for testing
    seed       : random seed for reproducible splits

    Returns
    -------
    X_train, Y_train, X_test, Y_test
        Lists of sentences and integer labels (1 = claim, 0 = not claim).
    """
    # Discover all document IDs (files that have both .json and .txt)
    json_files = sorted(glob.glob(os.path.join(data_dir, "nodeset*.json")))

    all_docs = []  # list of (sentences, labels) per document
    skipped = 0

    for jf in json_files:
        base = jf.replace(".json", "")
        tf = base + ".txt"
        if not os.path.exists(tf):
            skipped += 1
            continue
        sents, labels = _parse_single_document(jf, tf)
        if sents:
            all_docs.append((sents, labels))

    print(f"[araucaria] Parsed {len(all_docs)} documents  "
          f"(skipped {skipped} without matching .txt)")

    # --- Document-level train/test split ---
    rng = random.Random(seed)
    indices = list(range(len(all_docs)))
    rng.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_ratio))
    train_indices = set(indices[:split_idx])
    test_indices = set(indices[split_idx:])

    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for i, (sents, labels) in enumerate(all_docs):
        if i in train_indices:
            X_train.extend(sents)
            Y_train.extend(labels)
        else:
            X_test.extend(sents)
            Y_test.extend(labels)

    # --- Print statistics ---
    total_claims_train = sum(Y_train)
    total_claims_test = sum(Y_test)
    print(f"[araucaria] TRAIN: {len(X_train)} sentences  "
          f"({total_claims_train} claims, "
          f"{len(X_train) - total_claims_train} non-claims)  "
          f"ratio = 1:{(len(X_train) - total_claims_train) / max(total_claims_train, 1):.1f}")
    print(f"[araucaria]  TEST: {len(X_test)} sentences  "
          f"({total_claims_test} claims, "
          f"{len(X_test) - total_claims_test} non-claims)  "
          f"ratio = 1:{(len(X_test) - total_claims_test) / max(total_claims_test, 1):.1f}")
    print(f"[araucaria] Documents: {len(train_indices)} train, "
          f"{len(test_indices)} test  (no overlap)")

    return X_train, Y_train, X_test, Y_test
