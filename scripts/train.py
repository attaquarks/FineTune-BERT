import os
import torch
from transformers import BertTokenizerFast, Trainer, TrainingArguments
from models.bert_qa import ModifiedBertForQA
from utils.data_loader import load_subject_data
from utils.qa_utils import prepare_train_features
from config import *
import sys 
from config import SUBJECTS 


def train_with_trainer(subject):
    """Trains a ModifiedBertForQA model for a given subject using Hugging Face Trainer."""
    print(f"Starting training for subject: {subject}")

    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_CONFIG["bert_model"])
    model = ModifiedBertForQA.from_pretrained(MODEL_CONFIG["bert_model"])

    # Load and prepare dataset
    dataset = load_subject_data(subject)
    tokenized_dataset = dataset.map(
        lambda x: prepare_train_features(x, tokenizer, MODEL_CONFIG["max_length"], MODEL_CONFIG["doc_stride"]),
        batched=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./models/{subject}_bert_qa_output",
        per_device_train_batch_size=MODEL_CONFIG["batch_size"],
        num_train_epochs=MODEL_CONFIG["epochs"],
        learning_rate=MODEL_CONFIG["learning_rate"],
        logging_dir=f"./models/{subject}_bert_qa_logs",
        save_strategy="epoch",
        logging_steps=10
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train
    trainer.train()
    
    # Save model and tokenizer
    model.save_pretrained(f"./models/{subject}_bert_qa")
    tokenizer.save_pretrained(f"./models/{subject}_bert_qa")

if __name__ == '__main__':
    print("Starting training for all subjects listed in config.py...")
    
    if "SUBJECTS" not in globals() or not SUBJECTS:
        print("Error: SUBJECTS list not defined or empty in config.py. Cannot train any models.")
        sys.exit(1) # Exit if no subjects are defined

    # Loop through each subject and train the model
    for subject in SUBJECTS:
        print(f"\n--- Training model for subject: {subject} ---")
        try:
            # Call the training function for the current subject
            # Pass the subject name (converted to lowercase for consistency in paths)
            train_with_trainer(subject.lower())
            print(f"--- Training for {subject} completed successfully ---")
        except Exception as e:
            print(f"An error occurred while training model for subject {subject}: {e}")
            print(f"Skipping training for {subject} and moving to the next subject.")

    print("\n--- Training process finished for all subjects ---")
    print("You can now use the extractive model for all trained subjects in the inference script.")
