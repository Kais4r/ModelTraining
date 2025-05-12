from datasets import load_from_disk

# Load previously saved dataset
dataset = load_from_disk("./data/dataset/small_squad_v2")

# Define preprocessing function
def preprocess(example):
    return {
        "input_text": example["context"],
        "target_text": example["question"] + " [SEP] " + example["answers"]["text"][0] if example["answers"]["text"] else "No answer"
    }

# Apply preprocessing to all splits
preprocessed_dataset = dataset.map(preprocess)

# Save the preprocessed dataset
preprocessed_dataset.save_to_disk("./data/preprocessed_small_squad_v2")