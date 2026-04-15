import torch
import transformers
import evaluate

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available())
print("has _utils:", hasattr(torch, "_utils"))
print(evaluate.__version__)

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_ds = ds.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

from datasets import load_dataset
from transformers import AutoTokenizer
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_ds = ds.map(tokenize_function, batched=True)
tokenized_ds