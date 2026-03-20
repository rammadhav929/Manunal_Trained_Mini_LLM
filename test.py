from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading your custom model...")

tokenizer = AutoTokenizer.from_pretrained("./custom_llm")
model = AutoModelForCausalLM.from_pretrained("./custom_llm")

print("\n=== MODEL DEBUG INFO ===")
print("Tokenizer model_max_length:", tokenizer.model_max_length)
print("Model n_positions:", model.config.n_positions)
print("Vocab size:", model.config.vocab_size)
print("Embedding matrix:", model.transformer.wte.weight.shape)

def llm(prompt):

    print("\n=== PROMPT DEBUG ===")

    tokens = tokenizer.encode(prompt)

    print("RAW TOKEN COUNT:", len(tokens))
    print("MAX TOKEN ID:", max(tokens))

    max_allowed = model.config.n_positions - 120
    print("SAFE INPUT LIMIT:", max_allowed)

    if len(tokens) > max_allowed:
        print("⚠️ TRUNCATING PROMPT")
        tokens = tokens[-max_allowed:]

    print("FINAL TOKEN COUNT:", len(tokens))

    input_ids = torch.tensor([tokens])

    print("INPUT SHAPE:", input_ids.shape)

    # Check token id safety
    if max(tokens) >= model.config.vocab_size:
        print("❌ TOKENIZER MISMATCH DETECTED")
        raise ValueError("Token id exceeds embedding vocab size")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return text

print(llm("what is a backend developer."))
