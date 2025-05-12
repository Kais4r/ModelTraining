from datasets import load_dataset, DatasetDict

# Load the full dataset and store cache in a custom folder
squad_v2 = load_dataset("squad_v2", cache_dir="./data/download")

# Shuffle and reduce to 10,000 samples from the 'train' split
small_dataset = squad_v2["train"].shuffle(seed=42).select(range(10000))

# Split into train, validation, test
split_dataset = small_dataset.train_test_split(test_size=0.2, seed=42)
val_test = split_dataset['test'].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    'train': split_dataset['train'],
    'validation': val_test['train'],
    'test': val_test['test']
})

# Save processed dataset to a custom folder
output_folder = "./data/dataset/small_squad_v2"
dataset.save_to_disk(output_folder)