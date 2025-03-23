import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import json

def setup_model():
    """Setup Deepseek model for inference."""
    print("Loading Deepseek model...")
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def create_cluster_prompt(titles):
    """Create prompt for cluster summarization."""
    titles_str = '\n'.join([f"- {title}" for title in titles])
    return f"""You are a research assistant helping to analyze machine learning papers.
    Below are titles of related research papers. Please provide:
    1. A short theme that connects these papers (1 sentence)
    2. Key research directions or trends (2-3 bullet points)
    
    Papers:
    {titles_str}
    
    Response:"""

def generate_summary(model, tokenizer, prompt):
    """Generate summary using Deepseek model."""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_novelty_score(embedding, cluster_embeddings):
    """Compute novelty score as average distance to other papers in cluster."""
    distances = euclidean_distances([embedding], cluster_embeddings)[0]
    return float(np.mean(distances))

def main():
    data_path = "data"
    
    # Load data
    df = pd.read_csv(os.path.join(data_path, 'papers_with_clusters.csv'))
    embeddings = np.load(os.path.join(data_path, 'embeddings.npy'))
    print(f"Loaded {len(df)} papers with {df['cluster'].nunique()} clusters")
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Generate summaries for each cluster
    cluster_summaries = {}
    for cluster in tqdm(df['cluster'].unique(), desc="Summarizing clusters"):
        if cluster >= 0:  # Skip noise cluster (-1)
            cluster_papers = df[df['cluster'] == cluster]
            prompt = create_cluster_prompt(cluster_papers['title'].tolist())
            summary = generate_summary(model, tokenizer, prompt)
            cluster_summaries[str(cluster)] = summary
    
    # Save summaries
    with open(os.path.join(data_path, 'cluster_summaries.json'), 'w') as f:
        json.dump(cluster_summaries, f, indent=2)
    
    # Compute novelty scores
    print("\nComputing novelty scores...")
    novelty_scores = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cluster = row['cluster']
        if cluster >= 0:
            cluster_papers = df[df['cluster'] == cluster]
            cluster_embeddings = embeddings[cluster_papers.index]
            score = compute_novelty_score(embeddings[idx], cluster_embeddings)
        else:
            score = 1.0  # High novelty for outliers
        novelty_scores.append(score)
    
    df['novelty_score'] = novelty_scores
    
    # Flag standout papers (top 10% novelty score per cluster)
    df['is_novel'] = False
    for cluster in df['cluster'].unique():
        if cluster >= 0:
            cluster_mask = df['cluster'] == cluster
            threshold = df[cluster_mask]['novelty_score'].quantile(0.9)
            df.loc[cluster_mask & (df['novelty_score'] >= threshold), 'is_novel'] = True
    
    # Save results
    df.to_csv(os.path.join(data_path, 'papers_analyzed.csv'), index=False)
    
    # Display novel papers
    novel_papers = df[df['is_novel']].sort_values('novelty_score', ascending=False)
    print(f"\nFound {len(novel_papers)} novel papers")
    print("\nTop 5 most novel papers:")
    print(novel_papers[['title', 'cluster', 'novelty_score']].head())

if __name__ == "__main__":
    main() 