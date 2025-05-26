from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import evaluate

from reranking.config import *
from reranking.data_preprocessing import get_training_data

# Load pretrained reranker model and tokenizer
model_name = CROSS_ENCODER_PRETRAINED_MODEL_MINILM_MARCO
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(training_data):
    return tokenizer(training_data["question"], training_data["snippet"], truncation=True, padding="max_length")

training_data = get_training_data(DATA_PATH)


tokenized_training_data = training_data.map(preprocess, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    preds = p.predictions
    preds = np.squeeze(preds)
    return accuracy.compute(predictions=preds, references=p.label_ids)


training_args = TrainingArguments(
    output_dir=RERANKER_PATH,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_data,
    eval_dataset=tokenized_training_data.select(range(100)),  # Optional: use subset for eval
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(RERANKER_PATH)
tokenizer.save_pretrained(RERANKER_PATH)