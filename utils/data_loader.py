import json
from datasets import Dataset
from pathlib import Path

def load_subject_data(subject_name):
    """
    Loads subject-specific QA data from a JSON file into a Hugging Face Dataset.

    Args:
        subject_name: The name of the subject (e.g., "science", "literature").
                       Assumes the data file is named f"data/{subject_name}.json".

    Returns:
        A Hugging Face Dataset object.
        Expected JSON format: list of {"context":..., "question":..., "answers": {"text":..., "answer_start":...}}
    """
    path = Path(f"data/{subject_name}.json")
    if not path.exists():
        raise FileNotFoundError(f"Data file not found for subject '{subject_name}': {path}")
        
    with open(path, "r") as f:
        data = json.load(f)
        
    # Check if data is in the expected list format
    if not isinstance(data, list):
         raise TypeError(f"Data file for subject '{subject_name}' is not in the expected list format.")
         
    # Basic check for expected keys in the first example if list is not empty
    if data and not all(k in data[0] for k in ["context", "question", "answers"]):
         print(f"Warning: Data file for subject '{subject_name}' may not have expected keys (context, question, answers).")
         
    # Basic check for the structure of the 'answers' field
    if data and "answers" in data[0] and not isinstance(data[0]["answers"], dict):
         raise TypeError(f"Data file for subject '{subject_name}' has an 'answers' field that is not a dictionary.")
         
    if data and "answers" in data[0] and isinstance(data[0]["answers"], dict) and not all(k in data[0]["answers"] for k in ["text", "answer_start"]):
         print(f"Warning: 'answers' field in data file for subject '{subject_name}' may not have expected keys (text, answer_start).")

    return Dataset.from_list(data) 