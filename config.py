import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Available subjects
SUBJECTS = ["science", "literature", "computation"]

# Model configurations
MODEL_CONFIG = {
    "bert_model": "bert-base-uncased",
    "t5_model": "t5-small",
    "max_length": 384,
    "doc_stride": 128,
    "batch_size": 8,
    "epochs": 3,
    "learning_rate": 3e-5
}

# Legacy configurations (kept for backward compatibility)
BERT_MODEL_NAME = MODEL_CONFIG["bert_model"]
MAX_SEQ_LENGTH = MODEL_CONFIG["max_length"]
BATCH_SIZE = MODEL_CONFIG["batch_size"]
LEARNING_RATE = MODEL_CONFIG["learning_rate"]
NUM_EPOCHS = MODEL_CONFIG["epochs"]

# Training configurations
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42 