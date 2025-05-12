from transformers import BartTokenizer, BartForConditionalGeneration

# Load the fine-tuned model and tokenizer
model_dir = "./model/bart-base-squadv2-final"
tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)
model.eval()

print("BART QA Model Ready. Type 'exit' to quit.\n")

while True:
    # Get user input
    context = input("Enter context: ").strip()
    if context.lower() == "exit":
        break

    # Prepare input text
    input_text = f"{context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output
    output_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Display the generated question and answer
    print(f"Generated Output: {output}\n")