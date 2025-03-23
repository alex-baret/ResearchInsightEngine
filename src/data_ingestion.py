import arxiv
import pandas as pd
from datetime import datetime, timedelta
import os
from tqdm import tqdm

def setup_directory():
    """Create and return the path to the data directory."""
    base_path = "data"
    os.makedirs(base_path, exist_ok=True)
    return base_path

def format_paper_title(title):
    """Clean and format paper title."""
    return title.strip().replace('\n', ' ').replace('  ', ' ')

def fetch_papers(days=7, category='cs.LG'):
    """Fetch papers from arXiv published in the last n days."""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    print(f"Fetching papers from {start_date.date()} to {end_date.date()}")
    
    # Create search query
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=1000,  # We'll filter by date later
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Fetch and process papers
    papers = []
    for result in tqdm(search.results(), desc="Fetching papers"):
        # Check if paper is within date range
        if start_date <= result.published <= end_date:
            paper = {
                'id': result.entry_id.split('/')[-1],
                'title': format_paper_title(result.title),
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'date': result.published.strftime('%Y-%m-%d'),
                'categories': result.categories,
                'comment': result.comment if result.comment else '',
                'pdf_url': result.pdf_url
            }
            papers.append(paper)
    
    return papers

def main():
    # Setup
    data_path = setup_directory()
    
    # Fetch papers
    papers = fetch_papers()
    print(f"\nFound {len(papers)} papers")
    
    # Save to CSV
    df = pd.DataFrame(papers)
    output_path = os.path.join(data_path, 'recent_papers.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved papers to: {output_path}")
    
    # Display sample
    print("\nSample of collected papers:")
    print(df[['title', 'date']].head())

if __name__ == "__main__":
    main() 