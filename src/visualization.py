import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# Page config
st.set_page_config(
    page_title="Research Insight Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
data_path = "data"
df = pd.read_csv(os.path.join(data_path, 'papers_analyzed.csv'))
with open(os.path.join(data_path, 'cluster_summaries.json'), 'r') as f:
    cluster_summaries = json.load(f)

# Title
st.title("📚 Research Insight Engine")
st.write(f"Analyzing {len(df)} papers from arXiv cs.LG")

# Sidebar filters
st.sidebar.title("Filters")
selected_cluster = st.sidebar.selectbox(
    "Select Cluster",
    options=sorted([c for c in df['cluster'].unique() if c >= 0])
)

show_novel = st.sidebar.checkbox("Show Novel Papers Only", False)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Cluster visualization
    st.subheader("Paper Clusters")
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data=['title'],
        title='Paper Landscape',
        width=800,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Cluster summary
    st.subheader(f"Cluster {selected_cluster} Summary")
    st.write(cluster_summaries.get(str(selected_cluster), "No summary available"))

# Paper list
st.subheader("Papers")
filtered_df = df[df['cluster'] == selected_cluster]
if show_novel:
    filtered_df = filtered_df[filtered_df['is_novel']]

for _, paper in filtered_df.sort_values('novelty_score', ascending=False).iterrows():
    with st.expander(f"{paper['title']} {'🌟' if paper['is_novel'] else ''}"):
        st.write(f"**Authors:** {paper['authors']}")
        st.write(f"**Date:** {paper['date']}")
        st.write(f"**Abstract:** {paper['abstract']}")
        st.write(f"**Novelty Score:** {paper['novelty_score']:.3f}")
        if paper['pdf_url']:
            st.markdown(f"[📄 PDF]({paper['pdf_url']})")

# Search functionality
st.sidebar.subheader("Search")
search_query = st.sidebar.text_input("Search papers by title or abstract")

if search_query:
    st.subheader("Search Results")
    mask = df['title'].str.contains(search_query, case=False) | \
           df['abstract'].str.contains(search_query, case=False)
    search_results = df[mask].sort_values('novelty_score', ascending=False)
    
    for _, paper in search_results.iterrows():
        with st.expander(f"{paper['title']} {'🌟' if paper['is_novel'] else ''}"):
            st.write(f"**Cluster:** {paper['cluster']}")
            st.write(f"**Authors:** {paper['authors']}")
            st.write(f"**Abstract:** {paper['abstract']}")
            st.write(f"**Novelty Score:** {paper['novelty_score']:.3f}")
            if paper['pdf_url']:
                st.markdown(f"[📄 PDF]({paper['pdf_url']})") 