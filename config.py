import os
import torch
import random
import numpy as np

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset Config
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_DIR = os.path.join(DATA_DIR, "ml-latest-small")

# File Paths for Artifacts
MF_MODEL_PATH = os.path.join(DATA_DIR, "mf_model.pkl")
MINILM_BASE_PATH = os.path.join(DATA_DIR, "minilm_base_embeddings.pkl")
MINILM_FT_PATH = os.path.join(DATA_DIR, "minilm_finetuned_embeddings.pkl")
QWEN_BASE_PATH = os.path.join(DATA_DIR, "qwen_base_embeddings.pkl")
QWEN_FT_PATH = os.path.join(DATA_DIR, "qwen_finetuned_embeddings.pkl")
DATA_SPLIT_PATH = os.path.join(DATA_DIR, "data_split.pkl") # To save train/test split for evaluation

# Model Params
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utility Functions ---
def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def normalize_scores(scores):
    """Min-max normalize scores to [0, 1] range."""
    if len(scores) == 0:
        return scores
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {k: 0.5 for k in scores}
    return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

def get_user_rated_items(df, user_id):
    """Get set of movie IDs that a user has rated."""
    return set(df[df['userId'] == user_id]['movieId'].values)