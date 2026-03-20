# pip install torch transformers datasets accelerate

import torch # Import the core PyTorch library for deep learning operations
from transformers import ( # Import Hugging Face components for model architecture and training
    GPT2Config, # Class to define the structural configuration of a GPT-2 model
    GPT2LMHeadModel, # The actual GPT-2 model class with a Language Modeling head
    Trainer, # High-level API to handle the training loop automatically
    TrainingArguments, # Configuration class for training hyperparameters and settings
    DataCollatorForLanguageModeling, # Handles batching and padding of text data for the model
    AutoTokenizer, # Tool to convert raw text into numerical tokens
    pipeline # Utility to easily run inference on trained models
)
from datasets import load_dataset # Utility to download and load datasets from Hugging Face hub

def main(): # Define main function to prevent recursive process spawning on Windows

    # 1. Load Dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1") # Download and load the WikiText-2 dataset

    # 2. Create Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Load the standard GPT-2 vocabulary and rules
    tokenizer.pad_token = tokenizer.eos_token # Set the End-of-Sentence token as the padding token

    def tokenize_function(examples): # Define helper to process raw text into model-ready numbers
        return tokenizer( # Return tokenized text with specific constraints
            examples["text"], # The raw text field from the dataset
            truncation=True, # Cut off text that exceeds the maximum length
            padding="max_length", # Add zeros to shorter text to reach a uniform length
            max_length=256 # Set the maximum sequence length for the model's memory
        )

    tokenized_datasets = dataset.map( # Apply the tokenizer to every row in the dataset
        tokenize_function, # Use the previously defined helper function
        batched=True, # Process rows in groups for better performance
        num_proc=2, # Use 2 CPU cores to speed up the processing
        remove_columns=["text"] # Delete the original raw text to save memory
    )

    # 3. Create GPT Model Config
    config = GPT2Config( # Define a custom, smaller architecture to save hardware resources
        vocab_size=tokenizer.vocab_size, # Ensure the model's "alphabet" matches the tokenizer
        n_positions=256, # Set the maximum sequence length the model can handle
        n_ctx=256, # Define the context window size for internal attention
        n_embd=384, # Set the hidden layer size (lower than original 768 for speed)
        n_layer=6, # Reduce number of layers (depth) to speed up CPU training
        n_head=6 # Reduce number of attention heads for lower computational cost
    )

    model = GPT2LMHeadModel(config) # Initialize a brand new GPT-2 model with the custom config

    device = "cuda" if torch.cuda.is_available() else "cpu" # Detect if a GPU is available, else use CPU
    model.to(device) # Move the model's math operations to the detected hardware

    print("Model running on:", next(model.parameters()).device) # Confirm which hardware is being used

    # 4. Data Collator
    data_collator = DataCollatorForLanguageModeling( # Prepare data batches for language modeling
        tokenizer=tokenizer, # Use our tokenizer to handle special tokens
        mlm=False # Set to False because GPT-2 is causal (predicts next word)
    )

    # 5. Training Arguments
    training_args = TrainingArguments( # Set the rules for the training process
        output_dir="./llm_model", # Folder where training results will be stored
        per_device_train_batch_size=8, # Number of samples processed at once per step
        per_device_eval_batch_size=8, # Number of samples used during the testing phase
        eval_strategy="epoch", # Run a test evaluation after every full pass of data
        num_train_epochs=2, # Number of times the model sees the entire dataset
        logging_dir="./logs", # Folder for training progress logs
        save_steps=500, # Save a backup of the model every 500 steps
        save_total_limit=2, # Keep only the 2 most recent model backups to save space
        fp16=torch.cuda.is_available(), # Use 16-bit math if GPU is available to speed up
        report_to="none" # Disable external logging services like WandB
    )

    # 6. Trainer
    trainer = Trainer( # Create the engine that will run the training
        model=model, # The GPT-2 model we want to train
        args=training_args, # The training rules we just defined
        train_dataset=tokenized_datasets["train"], # The data used for learning
        eval_dataset=tokenized_datasets["validation"], # The data used for testing
        data_collator=data_collator # The tool that batches the data
    )

    trainer.train() # Start the actual training process (takes time!)

    # 7. Save Model
    trainer.save_model("./custom_llm") # Save the learned weights to a local folder
    tokenizer.save_pretrained("./custom_llm") # Save the tokenizer rules to the same folder

    print("Training complete. Model saved to ./custom_llm") # Print final success message

if __name__ == "__main__": # Standard Python entry point check
    main() # Execute the main function
