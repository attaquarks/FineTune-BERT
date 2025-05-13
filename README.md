# SubjectQA - Interactive Question Answering System

SubjectQA is an interactive question answering system that uses advanced NLP models to provide accurate answers to questions based on given context. The system supports multiple subjects and can use either BERT or a hybrid T5 model for answering questions.

## Features

- Interactive command-line interface for question answering
- Support for multiple subjects
- Option to use either BERT or hybrid T5 model
- Context-based question answering
- Simple and intuitive user interface

## Requirements

- Python 3.7 or higher
- PyTorch (>=1.9.0)
- Transformers (>=4.11.0)
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SubjectQA.git
cd SubjectQA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the interactive QA system:
```bash
python interactive_qa.py
```

2. Follow the prompts to:
   - Select a subject from the available options
   - Enter your question
   - Provide the context for the question
   - Choose whether to use the hybrid (T5) model or not

3. The system will process your input and provide an answer.

4. Type 'quit' or 'exit' at any prompt to exit the program.

## Project Structure:
SubjectQA/
├── data/ # Data directory
├── models/ # Model implementations
├── scripts/ # Utility scripts
├── utils/ # Helper utilities
├── config.py # Configuration settings
├── interactive_qa.py # Main interactive interface
└── requirements.txt # Project dependencies


## Configuration

The `config.py` file contains the configuration settings for the project, including:
- Available subjects
- Model parameters
- Other system settings

## Notes

- Make sure all required directories and files are present before running the system
- The system requires proper model files to be present in the models directory
- For best results, provide clear and specific context for your questions

## License

This project is licensed under the MIT License.
