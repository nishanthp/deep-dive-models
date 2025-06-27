from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch
import numpy as np
import evaluate

# 1. Load Pretrained Model & Tokenizer
#model_name = "bert-base-uncased"
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Load IMDB Dataset
dataset = load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(1000))
test_data = dataset["test"].shuffle(seed=42).select(range(500))

# 3. Tokenize the Dataset
def preprocess(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_data = train_data.map(preprocess, batched=True)
test_data = test_data.map(preprocess, batched=True)

# 4. Set Format for PyTorch
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 5. Define Evaluation Metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels.cpu.numpy() if torch.is_tensor(labels) else labels)

# 6. Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="no",
    logging_dir="./logs",
    report_to="none"
)

# 7. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Train and Evaluate
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)

# 9. Try Prediction
sample_text = "This movie was absolutely fantastic!"
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label = "positive" if prediction == 1 else "negative"

print(f'Sample Review: "{sample_text}" → Sentiment: {label}')

