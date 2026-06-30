import os
import re
import numpy as np
import torch
from transformers import BertTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import Preprocessing.araucaria_dataset
import Preprocessing.aaec_dataset

# Optimize PyTorch CPU threads for parallel linear algebra
torch.set_num_threads(4)

print("--> Loading AraucariaDB and AAEC datasets...")
araucaria_train_docs, araucaria_train_labels, araucaria_test_docs, araucaria_test_labels = \
    Preprocessing.araucaria_dataset.getDataLabelledDocuments()

aaec_train_docs, aaec_train_labels, aaec_test_docs, aaec_test_labels = \
    Preprocessing.aaec_dataset.getDataLabelledDocuments()

# Merge documents and labels
train_docs = araucaria_train_docs + aaec_train_docs
train_labels = araucaria_train_labels + aaec_train_labels
test_docs = araucaria_test_docs + aaec_test_docs
test_labels = araucaria_test_labels + aaec_test_labels


def prepare_context_data(docs, labels):
    texts = []
    flat_labels = []
    for doc_sents, doc_lbls in zip(docs, labels):
        for j in range(len(doc_sents)):
            prev_sent = doc_sents[j-1].strip() if j > 0 else ""
            target_sent = doc_sents[j].strip()
            next_sent = doc_sents[j+1].strip() if j < len(doc_sents) - 1 else ""
            
            # Format text representation with context indicators
            formatted_text = f"Prev: {prev_sent} | Target: {target_sent} | Next: {next_sent}"
            texts.append(formatted_text)
            flat_labels.append(doc_lbls[j])
    return texts, flat_labels


print("--> Preparing formatted text inputs with context...")
train_texts, train_flat_labels = prepare_context_data(train_docs, train_labels)
test_texts, test_flat_labels = prepare_context_data(test_docs, test_labels)

print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")

# Load pre-trained tiny tokenizer and model for extremely fast CPU training
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
print(f"--> Loading pre-trained {MODEL_NAME} tokenizer...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

print("--> Tokenizing dataset (without padding for dynamic batch-collator speedup)...")
train_encodings = tokenizer(train_texts, truncation=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, max_length=128)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = Dataset(train_encodings, train_flat_labels)
test_dataset = Dataset(test_encodings, test_flat_labels)

print(f"--> Loading pre-trained {MODEL_NAME} sequence classifier...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Dynamic Padding Collator for 3x speedup on CPU
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=3e-5,
    per_device_train_batch_size=32,  # larger batch size for CPU speedup
    per_device_eval_batch_size=32,
    num_train_epochs=4,              # 4 epochs is fast and optimal for bert-tiny
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    use_cpu=True                     # force CPU training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("--> Starting fine-tuning loop on CPU...")
trainer.train()

print("--> Evaluating on held-out test set...")
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Print final classification report
test_preds_logits = trainer.predict(test_dataset).predictions
test_preds = np.argmax(test_preds_logits, axis=-1)

print("\n" + "=" * 60)
print("FINE-TUNED BERT-TINY CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(test_flat_labels, test_preds, target_names=["Not claim", "Claim"]))

# Save model and tokenizer
save_path = "./bert_claim_model"
print(f"--> Saving fine-tuned model and tokenizer to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Finished!")
