from typing import Dict, List, Tuple
import torch
import numpy as np
from transformers import BertTokenizerFast

def prepare_train_features(example, tokenizer, max_length, doc_stride):
    # Tokenize the inputs
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"  # Ensure we get PyTorch tensors
    )
    

    offset_mapping = tokenized.pop("offset_mapping")
    
    # The 'overflow_to_sample_mapping' is not needed for the model forward pass
    if "overflow_to_sample_mapping" in tokenized:
        tokenized.pop("overflow_to_sample_mapping")
    
    # Get the answer positions
    # Assumes example["answers"] is a dictionary like {"text": "...", "answer_start": ...}
    answers = example["answers"]
    start_char = answers["answer_start"]
    end_char = start_char + len(answers["text"])
    
    # Find the token indices for the answer within the first (and only) sequence
    sequence_id = tokenized.sequence_ids(0)
    
    # Find the start and end token indices for the answer in the context portion
    token_start_index = 0
    token_end_index = len(tokenized["input_ids"][0]) - 1

    
    # Find the first token of the context
    context_start_token_index = 0
    while sequence_id[context_start_token_index] != 1:
        context_start_token_index += 1

    # Find the last token of the context
    context_end_token_index = len(tokenized["input_ids"][0]) - 1
    while sequence_id[context_end_token_index] != 1:
        context_end_token_index -= 1
 
    # Find the tokenized answer span
    token_start_index = context_start_token_index
    token_end_index = context_end_token_index

    # Adjust token_start_index and token_end_index based on the character positions
    for i in range(context_start_token_index, context_end_token_index + 1):
        # Get the character offsets for the token
        start_char_token, end_char_token = offset_mapping[0][i]

        # If the answer starts within this token
        if start_char >= start_char_token and start_char < end_char_token:
            token_start_index = i

        # If the answer ends within this token
        if end_char > start_char_token and end_char <= end_char_token:
            token_end_index = i
            break # Found the end, can stop


    tokenized["start_positions"] = torch.tensor(token_start_index)
    tokenized["end_positions"] = torch.tensor(token_end_index)
    
    tokenized["input_ids"] = tokenized["input_ids"].squeeze(0)
    tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0)
    # If using token_type_ids, squeeze that too
    if "token_type_ids" in tokenized:
         tokenized["token_type_ids"] = tokenized["token_type_ids"].squeeze(0)

    return tokenized

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics for QA model."""
    exact_match = sum(pred == ref for pred, ref in zip(predictions, references)) / len(predictions)
    return {
        'exact_match': exact_match
    }

def postprocess_qa_predictions(
    predictions: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
    tokenizer
) -> List[str]:
    """Convert model predictions to answer strings."""
    answers = []
    for i in range(predictions.size(0)): # Iterate over batch
        start = start_positions[i].item() if start_positions.ndim > 0 else start_positions.item()
        end = end_positions[i].item() if end_positions.ndim > 0 else end_positions.item()
        
        input_ids = predictions # Assuming predictions are the input_ids batch
        
        if start >= 0 and end < input_ids.size(1) and start <= end:
             answer_tokens = input_ids[i][start:end+1]
             answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        else:
             answer = "" # Or a special token indicating no answer
             
        answers.append(answer)
        
    return answers 