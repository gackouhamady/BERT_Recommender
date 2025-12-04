import sys
import os
import pickle
import zipfile
import urllib.request
import shutil
from itertools import combinations
from collections import defaultdict
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
from cornac.data import Dataset
from cornac.models import PMF
from peft import LoraConfig, get_peft_model, TaskType

# Add root directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def download_data():
    if not os.path.exists(config.ML_DIR):
        print("📥 Downloading MovieLens dataset...")
        zip_path = os.path.join(config.DATA_DIR, "ml-latest-small.zip")
        urllib.request.urlretrieve(config.DATASET_URL, zip_path)
        
        print("📂 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.DATA_DIR)
        os.remove(zip_path)
        print("✅ Dataset ready!")
    else:
        print("✅ Dataset already exists!")

def load_and_process_data():
    ratings_df = pd.read_csv(os.path.join(config.ML_DIR, "ratings.csv"))
    movies_df = pd.read_csv(os.path.join(config.ML_DIR, "movies.csv"))
    tags_df = pd.read_csv(os.path.join(config.ML_DIR, "tags.csv"))
    
    # Text Content Creation
    print("🔧 Creating movie content dictionary...")
    movie_tags = tags_df.groupby('movieId')['tag'].apply(
        lambda x: ' '.join(x.astype(str).str.lower())
    ).to_dict()

    movie_content = {}
    for _, row in movies_df.iterrows():
        movie_id = row['movieId']
        title = row['title'].rsplit('(', 1)[0].strip()
        genres = row['genres'].replace('|', ' ').replace('(no genres listed)', '')
        tags = movie_tags.get(movie_id, '')
        content = f"{title} {genres} {tags}".strip()
        movie_content[movie_id] = content if content else "movie"
    
    # Train/Test Split
    print("🔀 Creating train/test split...")
    train_list = []
    test_list = []
    for user_id, group in ratings_df.groupby('userId'):
        if len(group) >= 5:
            user_train, user_test = train_test_split(group, test_size=0.2, random_state=config.SEED)
            train_list.append(user_train)
            test_list.append(user_test)
        else:
            train_list.append(group)
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    # ID Mappings
    unique_users = train_df['userId'].unique()
    unique_items = train_df['movieId'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}
    
    # Save Split and Mappings for Evaluation
    data_artifacts = {
        'train_df': train_df,
        'test_df': test_df,
        'movies_df': movies_df,
        'movie_content': movie_content,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item
    }
    with open(config.DATA_SPLIT_PATH, 'wb') as f:
        pickle.dump(data_artifacts, f)
        
    return train_df, movie_content, user_to_idx, item_to_idx, list(movie_content.keys())

def train_mf(train_df, user_to_idx, item_to_idx):
    print("🚀 Training PMF model...")
    train_uir = [
        (user_to_idx[row['userId']], item_to_idx[row['movieId']], row['rating'])
        for _, row in train_df.iterrows()
    ]
    train_set = Dataset.from_uir(train_uir, seed=config.SEED)
    
    mf_model = PMF(k=100, max_iter=200, learning_rate=0.005, lambda_reg=0.001, verbose=True, seed=config.SEED)
    mf_model.fit(train_set)
    
    with open(config.MF_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': mf_model, 'train_set': train_set}, f)
    print("✅ PMF model saved.")

def generate_embeddings(model, movie_content, movie_ids, save_path, batch_size=64):
    print(f"🧠 Generating embeddings to {os.path.basename(save_path)}...")
    movie_texts = [movie_content[mid] for mid in movie_ids]
    
    # Helper to encode in chunks
    embeddings = []
    chunk_size = 500
    for i in tqdm(range(0, len(movie_texts), chunk_size)):
        chunk = movie_texts[i:i+chunk_size]
        emb = model.encode(chunk, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(emb)
    
    embeddings_array = np.vstack(embeddings)
    emb_dict = {mid: emb for mid, emb in zip(movie_ids, embeddings_array)}
    
    with open(save_path, 'wb') as f:
        pickle.dump(emb_dict, f)

def prepare_training_pairs(train_df, movie_content):
    print("📊 Creating positive pairs for fine-tuning...")
    user_liked = train_df[train_df['rating'] >= 4.0].groupby('userId')['movieId'].apply(list).to_dict()
    pairs = []
    for uid, movies in user_liked.items():
        if len(movies) < 2: continue
        curr_pairs = list(combinations(movies, 2))
        if len(curr_pairs) > 10: curr_pairs = random.sample(curr_pairs, 10)
        for ma, mb in curr_pairs:
            if ma in movie_content and mb in movie_content:
                pairs.append((movie_content[ma], movie_content[mb]))
    
    if len(pairs) > 10000: pairs = random.sample(pairs, 10000)
    return pairs

def main():
    config.set_seed()
    download_data()
    train_df, movie_content, u2i, i2i, movie_ids = load_and_process_data()
    
    # 1. Train Matrix Factorization
    train_mf(train_df, u2i, i2i)
    
    training_pairs = prepare_training_pairs(train_df, movie_content)
    
    # 2. MiniLM Base
    print("📥 Processing MiniLM Base...")
    minilm = SentenceTransformer('all-MiniLM-L6-v2')
    generate_embeddings(minilm, movie_content, movie_ids, config.MINILM_BASE_PATH)
    
    # 3. MiniLM Fine-tuning
    print("🎯 Fine-tuning MiniLM...")
    train_examples = [InputExample(texts=[t1, t2]) for t1, t2 in training_pairs]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(minilm)
    minilm.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, show_progress_bar=True)
    
    # Save Model
    save_path = os.path.join(config.MODELS_DIR, "fine_tuned_minilm")
    minilm.save(save_path)
    shutil.make_archive(save_path, 'zip', save_path)
    # Generate FT Embeddings
    generate_embeddings(minilm, movie_content, movie_ids, config.MINILM_FT_PATH)
    
    # 4. Qwen Base (Optional based on GPU)
    if torch.cuda.is_available():
        print("📥 Processing Qwen3-4B Base...")
        torch.cuda.empty_cache()
        qwen = SentenceTransformer('Qwen/Qwen3-Embedding-4B', trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})
        generate_embeddings(qwen, movie_content, movie_ids, config.QWEN_BASE_PATH, batch_size=8)
        
        # 5. Qwen Fine-tuning with LoRA
        print("🎯 Fine-tuning Qwen with LoRA...")
        del qwen
        torch.cuda.empty_cache()
        
        qwen_ft = SentenceTransformer('Qwen/Qwen3-Embedding-4B', trust_remote_code=True, 
                                      model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True})
        
        # Apply LoRA
        base_model = qwen_ft._first_module().auto_model
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], 
                                 lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION)
        peft_model = get_peft_model(base_model, lora_config)
        qwen_ft._first_module().auto_model = peft_model
        
        train_examples = [InputExample(texts=[t1, t2]) for t1, t2 in training_pairs]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
        train_loss = losses.MultipleNegativesRankingLoss(qwen_ft)
        
        qwen_ft.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, show_progress_bar=True)
        
        # Save Adapter
        adapter_path = os.path.join(config.MODELS_DIR, "qwen_lora_adapters_v1")
        peft_model.save_pretrained(adapter_path)
        shutil.make_archive(adapter_path, 'zip', adapter_path)
        
        # Generate Qwen FT Embeddings
        generate_embeddings(qwen_ft, movie_content, movie_ids, config.QWEN_FT_PATH, batch_size=4)

if __name__ == "__main__":
    main()