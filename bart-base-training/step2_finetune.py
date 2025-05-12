from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load preprocessed dataset (with 'input_text' and 'target_text' fields)
preprocessed_dataset = load_from_disk("./data/preprocessed_small_squad_v2")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model.to(device)  # Move model to GPU if available

# Tokenize
def tokenize(example):
    model_inputs = tokenizer(example["input_text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = preprocessed_dataset.map(tokenize, batched=True)

# Training config
training_args = Seq2SeqTrainingArguments(
    output_dir="./model/bart-base-squadv2",
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=300,
    eval_steps=300,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_dir="./bart-base-training/logs",
    logging_steps=100,
    predict_with_generate=True,
    save_total_limit=3,
    no_cuda=False, 
    fp16=True, 
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer
)

trainer.train()

# Save the final model
trainer.save_model("./model/bart-base-squadv2-final")