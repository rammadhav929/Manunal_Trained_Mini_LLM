from transformers import AutoTokenizer, AutoModelForCausalLM   # Import tokenizer and causal language model loader
import torch                                                   # Import PyTorch for tensor and model execution

print("Loading your custom model...")                          # Inform user model loading started

tokenizer = AutoTokenizer.from_pretrained("./custom_llm")      # Load tokenizer files from local trained model folder
model = AutoModelForCausalLM.from_pretrained("./custom_llm")   # Load trained causal language model weights

print("\n=== MODEL DEBUG INFO ===")                             # Print debug section header

print("Tokenizer model_max_length:", tokenizer.model_max_length)   # Show tokenizer max supported sequence length
print("Model n_positions:", model.config.n_positions)               # Show model positional embedding limit (true context window)
print("Vocab size:", model.config.vocab_size)                       # Show total vocabulary size of model
print("Embedding matrix:", model.transformer.wte.weight.shape)      # Show embedding weight tensor dimensions

def llm(prompt):                                                # Define function to generate text from prompt

    print("\n=== PROMPT DEBUG ===")                             # Debug header for prompt analysis

    tokens = tokenizer.encode(prompt)                           # Convert input prompt text into token IDs

    print("RAW TOKEN COUNT:", len(tokens))                      # Print number of tokens in raw prompt
    print("MAX TOKEN ID:", max(tokens))                         # Print largest token index in prompt

    max_allowed = model.config.n_positions - 120                # Reserve space for generation tokens
    print("SAFE INPUT LIMIT:", max_allowed)                     # Show safe usable context length

    if len(tokens) > max_allowed:                               # If prompt exceeds safe limit
        print("⚠️ TRUNCATING PROMPT")                            # Warn about truncation
        tokens = tokens[-max_allowed:]                          # Keep only last tokens to fit context window

    print("FINAL TOKEN COUNT:", len(tokens))                    # Show token count after truncation

    input_ids = torch.tensor([tokens])                          # Convert token list into PyTorch tensor batch

    print("INPUT SHAPE:", input_ids.shape)                      # Show tensor shape (batch_size, seq_len)

    # Check token id safety
    if max(tokens) >= model.config.vocab_size:                  # Validate token IDs are within vocab range
        print("❌ TOKENIZER MISMATCH DETECTED")                  # Print mismatch warning
        raise ValueError("Token id exceeds embedding vocab size") # Stop execution if mismatch occurs

    with torch.no_grad():                                       # Disable gradient tracking for inference
        output = model.generate(                                # Generate continuation text
            input_ids=input_ids,                                # Provide input tokens
            max_new_tokens=100,                                 # Limit generated token length
            do_sample=True,                                     # Enable probabilistic sampling
            temperature=0.7,                                    # Control randomness of generation
            top_k=50,                                           # Restrict sampling to top-k tokens
            top_p=0.95,                                         # Use nucleus sampling threshold
            pad_token_id=tokenizer.eos_token_id                 # Set padding token to EOS token
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True) # Convert generated token IDs back to text

    return text                                                  # Return generated string output

print(llm("what is a backend developer."))                       # Run test generation with sample prompt
