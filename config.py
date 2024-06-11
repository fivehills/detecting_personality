import os
from dotenv import load_dotenv

load_dotenv()

# Configuration settings
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SEED = 218
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
DATA_URL = os.getenv("DATA_URL", "path/to/your/dataset.json")

