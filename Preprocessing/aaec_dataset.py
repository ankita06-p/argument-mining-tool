import os
import glob
import re
import nltk
from difflib import SequenceMatcher

# NLTK downloads
nltk.download('punkt', quiet=True)

# Path relative to the working directory (i.e. ArgumentMining/)
AAEC_DIR = os.path.join("Datasets", "Student essays", "ArgumentAnnotatedEssays-2.0")
BRAT_DIR = os.path.join(AAEC_DIR, "data", "brat-project-final")
SPLIT_CSV = os.path.join(AAEC_DIR, "train-test-split.csv")
SIMILARITY_THRESHOLD = 0.78


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
        if n_norm in s_norm or s_norm in n_norm:
            return True
        ratio = SequenceMatcher(None, s_norm, n_norm).ratio()
        if ratio >= SIMILARITY_THRESHOLD:
            return True
    return False


def _parse_single_document(txt_path: str, ann_path: str):
    """
    Parse one txt/ann pair and return (sentences, labels).
    """
    # Load raw text
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read().replace("\n", " ").strip()

    if not raw_text:
        return [], []

    # Segment sentences
    sentences = nltk.tokenize.sent_tokenize(raw_text)

    # Load annotations (Claims, MajorClaims, Premises)
    claims = []
    if os.path.exists(ann_path):
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("T"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        tag_info = parts[1].split()
                        tag_type = tag_info[0]
                        tag_text = parts[2]
                        if tag_type in ["MajorClaim", "Claim", "Premise"]:
                            claims.append(tag_text)

    # Label sentences
    labels = []
    for sent in sentences:
        if _sentence_matches_any_node(sent, claims):
            labels.append(1)
        else:
            labels.append(0)

    return sentences, labels


def getDataLabelledDocuments(
    data_dir: str = BRAT_DIR,
    split_csv: str = SPLIT_CSV,
):
    """
    Parse all AAEC documents and return document-level train/test splits based on train-test-split.csv.

    Returns
    -------
    train_docs, train_labels, test_docs, test_labels
        Lists of documents, where each document is a list of sentences/labels.
    """
    # 1. Parse train-test-split.csv
    split_map = {}
    if os.path.exists(split_csv):
        with open(split_csv, "r", encoding="utf-8") as f:
            content = f.read()
        # Find all patterns of "essayXXX";"TRAIN" or "TEST"
        pairs = re.findall(r'"(essay\d+)";"(TRAIN|TEST)"', content)
        for essay_id, split_set in pairs:
            split_map[essay_id] = split_set

    # 2. Parse all text files
    txt_files = sorted(glob.glob(os.path.join(data_dir, "essay*.txt")))

    train_docs, train_labels = [], []
    test_docs, test_labels = [], []

    for tf in txt_files:
        base = os.path.basename(tf).replace(".txt", "")
        ann_path = tf.replace(".txt", ".ann")
        
        sents, labels = _parse_single_document(tf, ann_path)
        if not sents:
            continue

        split_set = split_map.get(base, "TRAIN")  # default to TRAIN if not found
        if split_set == "TRAIN":
            train_docs.append(sents)
            train_labels.append(labels)
        else:
            test_docs.append(sents)
            test_labels.append(labels)

    # Print stats
    total_train_sents = sum(len(d) for d in train_docs)
    total_test_sents = sum(len(d) for d in test_docs)
    total_train_claims = sum(sum(l) for l in train_labels)
    total_test_claims = sum(sum(l) for l in test_labels)

    print(f"[aaec] Documents Loaded: {len(train_docs)} train, {len(test_docs)} test")
    print(f"[aaec] Total train sentences: {total_train_sents} ({total_train_claims} claims)")
    print(f"[aaec] Total test sentences: {total_test_sents} ({total_test_claims} claims)")

    return train_docs, train_labels, test_docs, test_labels
