import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, load_from_disk
import numpy as np
from tqdm import tqdm

# Load model and tokenizer
model_path = "./model/t5-small-squadv2-final"  # change to your actual path
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load your 10k split dataset
dataset = load_from_disk("./data/preprocessed_small_squad_v2")

# Use your train, validation, and test splits (make sure you've created this)
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Prepare examples from the dataset (context, question, and answer)
examples = []
references = []
for ex in test_dataset:  # Use test dataset for evaluation
    if len(ex["answers"]["text"]) == 0:
        continue
    context = ex["context"]
    question = ex["question"]
    answer = ex["answers"]["text"][0]
    input_text = context
    target_text = question + " [SEP] " + answer
    examples.append(input_text)
    references.append(target_text)

# Generate predictions
predictions = []
model.eval()
for input_text in tqdm(examples):
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=128)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predictions.append(output)

# Split predictions into question and answer
def split_pred(pred):
    parts = pred.split("[SEP]")
    if len(parts) == 2:
        question, answer = parts
    else:
        question, answer = pred, ""
    return question.strip(), answer.strip()

pred_qa = [split_pred(p) for p in predictions]
ref_qa = [split_pred(r) for r in references]

pred_questions, pred_answers = zip(*pred_qa)
ref_questions, ref_answers = zip(*ref_qa)

# Load metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

# Custom exact match
def exact_match(preds, refs):
    return np.mean([p.strip().lower() == r.strip().lower() for p, r in zip(preds, refs)])

# Custom token-level F1
def f1_score(preds, refs):
    def score(p, r):
        p_tokens = p.strip().lower().split()
        r_tokens = r.strip().lower().split()
        common = set(p_tokens) & set(r_tokens)
        if not common:
            return 0.0
        precision = len(common) / len(p_tokens)
        recall = len(common) / len(r_tokens)
        return 2 * (precision * recall) / (precision + recall)
    return np.mean([score(p, r) for p, r in zip(preds, refs)])

# Evaluate on answers (could repeat for questions too)
results = {
    "ROUGE": rouge.compute(predictions=pred_answers, references=ref_answers, use_stemmer=True),
    "BLEU": bleu.compute(predictions=pred_answers, references=[[r] for r in ref_answers]),
    "METEOR": meteor.compute(predictions=pred_answers, references=ref_answers),
    "BERTScore": bertscore.compute(predictions=pred_answers, references=ref_answers, lang="en"),
    "Exact Match": exact_match(pred_answers, ref_answers),
    "F1 Score": f1_score(pred_answers, ref_answers)
}

# Print results
print("\n--- Evaluation Metrics (Answers) ---")
# for key, value in results.items():
#     if isinstance(value, dict):
#         for subkey, subval in value.items():
#             if isinstance(subval, (list, np.ndarray)):
#                 print(f"{key} - {subkey}: {np.mean(subval):.4f}")
#             else:
#                 print(f"{key} - {subkey}: {subval:.4f}")
#     else:
#         print(f"{key}: {value:.4f}")

for key, value in results.items():
    if isinstance(value, dict):
        for subkey, subval in value.items():
            if isinstance(subval, (float, int)):
                print(f"{key} - {subkey}: {subval:.4f}")
            else:
                print(f"{key} - {subkey}: {subval}")
    else:
        print(f"{key}: {value:.4f}")
