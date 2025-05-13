#!/usr/bin/env python
import sys
import os

# Add the project root to sys.path to allow importing modules like scripts, models, utils
# This helps the script find your modules when run directly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

try:
    # Now you can import infer_single using its full path from the project root
    from scripts.infer import infer_single
    from config import SUBJECTS # Import SUBJECTS from config
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure you are running this script from the SubjectQA project root.")
    print("Also, make sure your scripts, models, and utils directories have __init__.py files.")
    sys.exit(1)

def main():
    print("--- SubjectQA Interactive Question Answering ---")
    print("Type 'quit' or 'exit' at any prompt to quit.")
    print(f"Available subjects: {', '.join(SUBJECTS)}")

    while True:
        try:
            subject = input(f"Enter subject ({'/'.join(SUBJECTS)}): ").lower().strip()
            if subject in ['quit', 'exit']:
                break
            if subject not in [s.lower() for s in SUBJECTS]:
                 print(f"Invalid subject. Please choose from: {', '.join(SUBJECTS)}")
                 continue

            question = input("Enter your question: ").strip()
            if question in ['quit', 'exit']:
                break
            if not question:
                print("Question cannot be empty.")
                continue

            context = input("Enter the context: ").strip()
            if context in ['quit', 'exit']:
                break
            if not context:
                print("Context cannot be empty.")
                continue

            use_hybrid_input = input("Use Hybrid (T5) model? (yes/no, default no): ").lower().strip()
            if use_hybrid_input in ['quit', 'exit']:
                break
            use_hybrid = use_hybrid_input == 'yes'

            print("\nGetting answer...")
            # Call the inference function
            answer = infer_single(question, context, subject, use_hybrid=use_hybrid)

            print(f"\nAnswer: {answer}")
            print("-" * 20) # Separator

        except EOFError: # Handle Ctrl+D
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Exiting interactive QA. Goodbye!")

if __name__ == "__main__":
    # Ensure the necessary directories and files exist before running
    required_dirs = ['models', 'utils', 'scripts', 'data']
    required_inits = ['models/__init__.py', 'utils/__init__.py', 'scripts/__init__.py']
    required_files = ['scripts/infer.py', 'models/bert_qa.py', 'models/hybrid_qa.py', 'utils/data_loader.py', 'utils/qa_utils.py', 'config.py']

    all_present = True
    for d in required_dirs:
        if not os.path.isdir(d):
            print(f"Error: Directory '{d}' not found. Please ensure the project structure is correct.")
            all_present = False
    for f in required_inits + required_files:
        if not os.path.exists(f):
             print(f"Error: File '{f}' not found. Please ensure the project structure is correct.")
             all_present = False

    if not all_present:
        print("\nCannot run interactive QA due to missing files/directories.")
        sys.exit(1)

    main()
