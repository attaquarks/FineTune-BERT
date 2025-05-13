import os
import torch
from transformers import BertTokenizerFast
from models.bert_qa import ModifiedBertForQA
from models.hybrid_qa import HybridQAModel
from config import *

def infer_single(question: str, context: str, subject: str, use_hybrid: bool = False, bert_model_path_template: str = None):
    """
    Performs inference for a single question-context pair using either a fine-tuned
    BERT-based model or a hybrid T5-based model.

    Args:
        question: The question string.
        context: The context string.
        subject: The subject domain, used to load the correct fine-tuned BERT model.
        use_hybrid: If True, uses the HybridQAModel (T5). Otherwise, uses ModifiedBertForQA.
        bert_model_path_template: Optional. A template for the BERT model path, 
                                   e.g., "models/{}_bert_qa_final". If None, defaults to 
                                   os.path.join(MODELS_DIR, f"{subject}_bert_qa_final").
    Returns:
        The predicted answer string.
    """
    print(f"Running inference for subject: {subject}, Hybrid: {use_hybrid}")
    print(f"Question: {question[:100]}...") # Print first 100 chars
    print(f"Context: {context[:200]}...")  # Print first 200 chars


    if use_hybrid:
        # Assuming HybridQAModel might take a T5 model name from MODEL_CONFIG in the future
        t5_model_name = MODEL_CONFIG.get("t5_model", "t5-small") 
        print(f"Using HybridQAModel with T5 base: {t5_model_name}")
        hybrid_model = HybridQAModel(t5_model_name=t5_model_name)
        answer = hybrid_model.generate_answer(question, context)
        print(f"Hybrid Model Answer: {answer}")
        return answer

    # Determine the path for the fine-tuned BERT model
    if bert_model_path_template:
        model_load_path = bert_model_path_template.format(subject)
    else:
        
        if "MODELS_DIR" in globals() and os.path.exists(MODELS_DIR):
             model_load_path = os.path.join(MODELS_DIR, f"{subject}_bert_qa_final")
        else:
             # Fallback if MODELS_DIR is not set or doesn't exist
             model_load_path = f"./models/{subject}_bert_qa_final"
             print(f"Warning: MODELS_DIR not found or does not exist in config. Using default path: {model_load_path}")
             
    
    print(f"Loading ModifiedBertForQA model and tokenizer from: {model_load_path}")
    
    if not os.path.exists(model_load_path) or not os.path.isdir(model_load_path):
        
        fallback_path = f"./models/{subject}_bert_qa" # As per user's infer code
        if os.path.exists(fallback_path) and os.path.isdir(fallback_path):
            print(f"Primary model path {model_load_path} not found or not a directory. Using fallback: {fallback_path}")
            model_load_path = fallback_path
        else:
            error_msg = f"Error: Model directory not found at {model_load_path} or {fallback_path}. Please train the model first."
            print(error_msg)
            return error_msg

    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_load_path)
        model = ModifiedBertForQA.from_pretrained(model_load_path)
        model.eval() # Ensure model is in evaluation mode
        if torch.cuda.is_available():
             model.to('cuda')
             print("Model moved to GPU.")
    except Exception as e:
        error_msg = f"Error loading model/tokenizer from {model_load_path}: {e}"
        print(error_msg)
        return error_msg

    print("Tokenizing input...")
    max_len = MODEL_CONFIG.get("max_length", 384)
    
    # Tokenize the question and context
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=max_len) # Added padding=True

    if torch.cuda.is_available():
         inputs = {k: v.to('cuda') for k, v in inputs.items()}
         
    print("Performing inference with ModifiedBertForQA...")
    with torch.no_grad():
        outputs = model(**inputs) 
        if "start_logits" not in outputs or "end_logits" not in outputs:
            error_msg = "Error: Model output does not contain 'start_logits' or 'end_logits'."
            print(error_msg)
            return error_msg
            
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        
        end_idx = end_idx + 1 
        
        input_ids = inputs["input_ids"]
        input_ids_len = input_ids.shape[1]
        
        if start_idx >= input_ids_len or end_idx > input_ids_len or start_idx >= end_idx:
            print(f"Warning: Invalid start/end logits. Start: {start_idx.item()}, End: {end_idx.item()-1}, Length: {input_ids_len}. Returning empty string.")
            answer = "" # Return empty string for invalid span
        else:
            # Decode the answer span
            answer_ids = input_ids[0][start_idx:end_idx]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    print(f"Extractive Model Answer: {answer}")
    return answer

if __name__ == '__main__':
    # Example usage:
    sample_question = "What is the chemical formula for water?"
    sample_context = "Water is a transparent, colorless liquid that forms the oceans, lakes, rivers, and rain. Its chemical formula is H2O. It is vital for all known forms of life."
    sample_subject = "science" # Should be one of the subjects in config.SUBJECTS

    print("\n--- Testing Extractive QA (ModifiedBertForQA) ---")
    
    extractive_answer = infer_single(sample_question, sample_context, sample_subject, use_hybrid=False)
    print(f"Final Extractive Answer for '{sample_question}': {extractive_answer}")

    print("\n--- Testing Generative QA (HybridQAModel - T5) ---")
    hybrid_answer = infer_single(sample_question, sample_context, sample_subject, use_hybrid=True)
    print(f"Final Hybrid Answer for '{sample_question}': {hybrid_answer}")
