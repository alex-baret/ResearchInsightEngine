import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import plotly.express as px
import os
from tqdm import tqdm

def load_papers(data_path):
    """Load papers from CSV."""
    return pd.read_csv(os.path.join(data_path, 'recent_papers.csv'))

def create_embeddings(texts):
    """Create embeddings using sentence-transformers."""
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    )
    return embeddings

def reduce_dimensions(embeddings):
    """Reduce dimensions using UMAP."""
    print("Reducing dimensions...")
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        random_state=42
    )
    return reducer.fit_transform(embeddings)

def cluster_papers(embeddings):
    """Cluster papers using HDBSCAN."""
    print("Clustering papers...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric='euclidean',
        prediction_data=True
    )
    return clusterer.fit_predict(embeddings)

def visualize_clusters(df, output_path):
    """Create and save cluster visualization."""
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['title'],
        title='Paper Clusters',
        width=1000,
        height=800
    )
    
    # Save as HTML
    fig.write_html(os.path.join(output_path, 'clusters.html'))
    print(f"Visualization saved to: {output_path}/clusters.html")

def main():
    data_path = "data"
    
    # Load papers
    df = load_papers(data_path)
    print(f"Loaded {len(df)} papers")
    
    # Create embeddings
    texts = df['title'] + " " + df['abstract']
    embeddings = create_embeddings(texts)
    np.save(os.path.join(data_path, 'embeddings.npy'), embeddings)
    
    # Reduce dimensions
    embeddings_2d = reduce_dimensions(embeddings)
    np.save(os.path.join(data_path, 'embeddings_2d.npy'), embeddings_2d)
    
    # Cluster papers
    cluster_labels = cluster_papers(embeddings)
    
    # Add results to dataframe
    df['cluster'] = cluster_labels
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    
    # Save results
    df.to_csv(os.path.join(data_path, 'papers_with_clusters.csv'), index=False)
    
    # Display cluster statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    print(f"\nFound {n_clusters} clusters")
    print(f"Noise points: {n_noise} ({n_noise/len(cluster_labels):.1%})")
    
    # Create visualization
    visualize_clusters(df, data_path)

if __name__ == "__main__":
    main() 