from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk

# Load preprocessed dataset (with 'input_text' and 'target_text' fields)
preprocessed_dataset = load_from_disk("./data/preprocessed_small_squad_v2")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize
def tokenize(example):
    model_inputs = tokenizer(example["input_text"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["target_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenized_data = tokenized_dataset.map(tokenize, batched=True)
tokenized_data = preprocessed_dataset.map(tokenize, batched=True)

# Training config
training_args = Seq2SeqTrainingArguments(
    output_dir="./model/t5-small-squadv2",
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=300,
    eval_steps=300,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    num_train_epochs=3,
    logging_dir="./t5-small-training/logs",
    logging_steps=100,
    predict_with_generate=True,
    save_total_limit=3,
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
trainer.save_model("./model/t5-small-squadv2-final")

# # # Save model from a specific check point
# checkpoint_dir = "./model/t5-small-squadv2/checkpoint-3000"

# # Load the model and tokenizer from the checkpoint
# model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
# tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
# save_dir = "./model/t5-small-squadv2-final"

# # Save the model and tokenizer
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)
# # This will create a directory named t5-small-squadv2-final containing the model and tokenizer files, which you can later load using from_pretrained.