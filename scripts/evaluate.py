import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from inference import recommend_mf_only, recommend_hybrid, load_resources

def calculate_ndcg(recommended, ground_truth, k):
    relevance = [1 if item in ground_truth else 0 for item in recommended[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_method(name, recommend_func, test_df, k=25, **kwargs):
    print(f"\n📊 Evaluating {name}...")
    precision, recall, ndcg = [], [], []
    
    # Required data from kwargs
    u2i = kwargs['user_to_idx']
    
    # Filter valid users
    valid_users = [u for u in test_df['userId'].unique() if u in u2i]
    
    for user_id in tqdm(valid_users):
        user_test = test_df[(test_df['userId'] == user_id) & (test_df['rating'] >= 4.0)]
        ground_truth = set(user_test['movieId'].values)
        if not ground_truth: continue
        
        recs = recommend_func(user_id, n=k, **kwargs)
        if not recs: continue
        
        rec_items = [r[0] for r in recs]
        hits = len(set(rec_items) & ground_truth)
        
        precision.append(hits / k)
        recall.append(hits / len(ground_truth))
        ndcg.append(calculate_ndcg(rec_items, ground_truth, k))
        
    return {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'ndcg': np.mean(ndcg)
    }

def main():
    data, mf_model, train_set = load_resources()
    
    results = {}
    common_args = {
        'mf_model': mf_model,
        'train_df': data['train_df'],
        'user_to_idx': data['user_to_idx'],
        'idx_to_item': data['idx_to_item']
    }
    
    # 1. Evaluate MF Only
    results['MF Only'] = evaluate_method(
        'MF Only', 
        recommend_mf_only, 
        data['test_df'], 
        train_set=train_set, 
        **common_args
    )
    
    # 2. Evaluate Embeddings
    embedding_files = {
        'MiniLM Base': config.MINILM_BASE_PATH,
        'MiniLM FT': config.MINILM_FT_PATH,
        'Qwen Base': config.QWEN_BASE_PATH,
        'Qwen FT': config.QWEN_FT_PATH
    }
    
    for name, path in embedding_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f: embs = pickle.load(f)
            results[name] = evaluate_method(
                name, 
                recommend_hybrid, 
                data['test_df'], 
                embeddings_dict=embs, 
                alpha=0.85, 
                **common_args
            )
        else:
            print(f"⚠️ Skipping {name} (File not found)")

    # Print Results
    print("\n" + "="*50)
    print(f"{'Method':<20} {'Precision':<10} {'Recall':<10} {'NDCG':<10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['precision']:.4f}     {res['recall']:.4f}     {res['ndcg']:.4f}")

    # Plotting
    methods = list(results.keys())
    precisions = [results[m]['precision'] for m in methods]
    recalls = [results[m]['recall'] for m in methods]
    ndcgs = [results[m]['ndcg'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label='Precision@25')
    ax.bar(x, recalls, width, label='Recall@25')
    ax.bar(x + width, ndcgs, width, label='NDCG@25')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_title("Recommendation Performance Comparison")
    
    plt.savefig("scores.png")
    print("\n✅ Chart saved to scores.png")

if __name__ == "__main__":
    main()