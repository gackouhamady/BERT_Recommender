import sys
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_resources():
    """Load necessary data and models."""
    with open(config.DATA_SPLIT_PATH, 'rb') as f:
        data = pickle.load(f)
    
    with open(config.MF_MODEL_PATH, 'rb') as f:
        mf_data = pickle.load(f)
        
    return data, mf_data['model'], mf_data['train_set']

def recommend_mf_only(user_id, mf_model, train_set, user_to_idx, idx_to_item, train_df, n=10):
    """MF Only Recommendation Strategy."""
    if user_id not in user_to_idx: return []
    user_idx = user_to_idx[user_id]
    
    if user_idx >= train_set.num_users: return []

    rated_items = config.get_user_rated_items(train_df, user_id)
    all_scores = mf_model.score(user_idx)

    candidates = []
    for item_idx, score in enumerate(all_scores):
        if item_idx in idx_to_item:
            movie_id = idx_to_item[item_idx]
            if movie_id not in rated_items:
                candidates.append((movie_id, float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]

def recommend_hybrid(user_id, embeddings_dict, mf_model, train_df, user_to_idx, idx_to_item, n=10, alpha=0.85):
    """Hybrid Recommendation Strategy."""
    if user_id not in user_to_idx: return []
    user_idx = user_to_idx[user_id]
    rated_items = config.get_user_rated_items(train_df, user_id)

    # Content Profile Calculation
    user_ratings = train_df[train_df['userId'] == user_id][['movieId', 'timestamp', 'rating']].copy()
    if len(user_ratings) == 0: return []

    # Recency Weighting
    min_ts, max_ts = user_ratings['timestamp'].min(), user_ratings['timestamp'].max()
    user_ratings['norm_ts'] = (user_ratings['timestamp'] - min_ts) / (max_ts - min_ts) if max_ts > min_ts else 1.0
    user_ratings['weight'] = np.exp(user_ratings['norm_ts']) * (user_ratings['rating'] / 5.0)

    # Build Vectors
    rated_embs = []
    weights = []
    for _, row in user_ratings.iterrows():
        if row['movieId'] in embeddings_dict:
            rated_embs.append(embeddings_dict[row['movieId']])
            weights.append(row['weight'])
            
    if not rated_embs: return []
    rated_matrix = np.vstack(rated_embs)
    weights = np.array(weights)

    # MF Scores
    all_mf_scores = mf_model.score(user_idx)
    mf_scores = {idx_to_item[i]: s for i, s in enumerate(all_mf_scores) if i in idx_to_item}
    mf_scores_norm = config.normalize_scores(mf_scores)

    # Candidate calculation
    candidate_ids = []
    candidate_embs = []
    for mid in mf_scores.keys():
        if mid not in rated_items and mid in embeddings_dict:
            candidate_ids.append(mid)
            candidate_embs.append(embeddings_dict[mid])
            
    if not candidate_embs: return []
    candidate_matrix = np.vstack(candidate_embs)

    # Similarity
    sims = cosine_similarity(candidate_matrix, rated_matrix)
    weighted_sims = sims @ weights
    content_scores = weighted_sims / weights.sum()
    
    content_scores_dict = {mid: s for mid, s in zip(candidate_ids, content_scores)}
    content_scores_norm = config.normalize_scores(content_scores_dict)

    # Fusion
    final_scores = []
    for mid in content_scores_norm:
        s_mf = mf_scores_norm.get(mid, 0)
        s_cont = content_scores_norm.get(mid, 0)
        final = alpha * s_mf + (1 - alpha) * s_cont
        final_scores.append((mid, final))
        
    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores[:n]

def run_demo():
    print("🎬 DEMO: Running Sample Recommendations")
    data, mf_model, train_set = load_resources()
    train_df = data['train_df']
    movies_df = data['movies_df']
    
    # Load one embedding set for demo (e.g., Qwen Base if available, else MiniLM)
    if os.path.exists(config.QWEN_BASE_PATH):
        with open(config.QWEN_BASE_PATH, 'rb') as f: embeddings = pickle.load(f)
        emb_name = "Qwen Base"
    else:
        with open(config.MINILM_BASE_PATH, 'rb') as f: embeddings = pickle.load(f)
        emb_name = "MiniLM Base"

    # Sample user
    sample_user = train_df['userId'].value_counts().index[0]
    print(f"\n👤 User ID: {sample_user}")
    
    print("\n🎯 MF Only Recommendations:")
    recs = recommend_mf_only(sample_user, mf_model, train_set, data['user_to_idx'], data['idx_to_item'], train_df, n=5)
    for mid, score in recs:
        title = movies_df[movies_df['movieId'] == mid]['title'].values[0]
        print(f"  - {title} ({score:.3f})")

    print(f"\n🎯 Hybrid ({emb_name}) Recommendations:")
    recs = recommend_hybrid(sample_user, embeddings, mf_model, train_df, data['user_to_idx'], data['idx_to_item'], n=5)
    for mid, score in recs:
        title = movies_df[movies_df['movieId'] == mid]['title'].values[0]
        print(f"  - {title} ({score:.3f})")

if __name__ == "__main__":
    run_demo()