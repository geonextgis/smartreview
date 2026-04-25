"""
SmartReview Utilities
Utility functions for embedding calculation, similarity analysis, and paper export.
"""

import numpy as np
import pandas as pd
import pickle
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(query_embedding, paper_embeddings_dict):
    """
    Calculate cosine similarity between query embedding and all paper embeddings.
    
    Args:
        query_embedding: numpy array of shape (embedding_dim,)
        paper_embeddings_dict: dictionary with paper indices as keys and embeddings as values
    
    Returns:
        list of tuples (paper_index, similarity_score) sorted by similarity (descending)
    """
    similarities = []
    
    for idx, paper_emb in paper_embeddings_dict.items():
        # Reshape for cosine_similarity function
        sim = cosine_similarity(
            query_embedding.reshape(1, -1), 
            paper_emb.reshape(1, -1)
        )[0][0]
        similarities.append((idx, sim))
    
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


def get_top_k_papers(similarities, k=100):
    """
    Extract top k papers from similarity scores.
    
    Args:
        similarities: list of tuples (paper_index, similarity_score)
        k: number of top papers to return
    
    Returns:
        list of top k papers (first k elements)
    """
    return similarities[:min(k, len(similarities))]


def create_top_k_dataframe(top_k_papers, data, summary):
    """
    Create a dataframe for top k papers with all information from the main data.
    
    Args:
        top_k_papers: list of (paper_idx, similarity) tuples
        data: main pandas DataFrame from Web of Science
        summary: dictionary with (title, abstract) pairs
    
    Returns:
        pandas DataFrame with rank, similarity, and all original columns
    """
    top_k_data = []
    
    for rank, (paper_idx, similarity) in enumerate(top_k_papers, 1):
        # Get the row from the main data dataframe
        row_data = data.iloc[paper_idx].to_dict()
        # Add rank and similarity score
        row_with_similarity = {
            "Rank": rank,
            "Similarity_Score": similarity
        }
        # Add all original data columns
        row_with_similarity.update(row_data)
        top_k_data.append(row_with_similarity)
    
    # Create dataframe sorted by rank
    top_k_df = pd.DataFrame(top_k_data)
    top_k_df = top_k_df.sort_values(by=["Rank"], ascending=True)
    
    return top_k_df


def save_top_k_papers(top_k_df, output_dir="data", k=100):
    """
    Save top k papers to CSV and Excel formats.
    
    Args:
        top_k_df: DataFrame containing top k papers
        output_dir: directory to save files
        k: number of papers (for naming)
    
    Returns:
        dict with file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, f"top_{k}_papers_complete.csv")
    excel_file = os.path.join(output_dir, f"top_{k}_papers_complete.xlsx")
    
    # Save to CSV
    top_k_df.to_csv(csv_file, index=False, encoding="utf-8")
    
    # Save to Excel
    try:
        top_k_df.to_excel(excel_file, index=False, engine="openpyxl")
    except ImportError:
        print("Warning: openpyxl not available, Excel file not created")
        excel_file = None
    
    return {
        "csv": csv_file,
        "excel": excel_file,
        "rows": len(top_k_df),
        "columns": len(top_k_df.columns)
    }


def create_bibtex_entry(rank, paper_data):
    """
    Create a BibTeX entry for a paper.
    
    Args:
        rank: Paper rank
        paper_data: Dictionary containing paper information
    
    Returns:
        BibTeX entry as a string
    """
    # Create a unique citation key based on first author's last name and year
    authors = paper_data.get('Authors', '')
    year = paper_data.get('Publication Year', '')
    title = paper_data.get('Article Title', '')
    
    # Extract first author's last name
    if pd.notna(authors) and len(str(authors)) > 0:
        first_author = str(authors).split(';')[0].split(',')[0].strip()
    else:
        first_author = 'Unknown'
    
    # Create citation key
    year_str = str(int(year)) if pd.notna(year) else '2000'
    citation_key = f"{first_author.replace(' ', '')}_{year_str}_{rank}"
    
    # Build BibTeX entry
    bibtex = f"@article{{{citation_key},\n"
    
    # Add fields
    if pd.notna(authors):
        # Clean up author names for BibTeX format
        author_str = str(authors).replace(';', ' and ')
        bibtex += f"  author = {{{author_str}}},\n"
    
    if pd.notna(title):
        bibtex += f"  title = {{{title}}},\n"
    
    if pd.notna(paper_data.get('Source Title')):
        bibtex += f"  journal = {{{paper_data.get('Source Title')}}},\n"
    
    if pd.notna(year):
        bibtex += f"  year = {{{year_str}}},\n"
    
    if pd.notna(paper_data.get('Volume')):
        bibtex += f"  volume = {{{paper_data.get('Volume')}}},\n"
    
    if pd.notna(paper_data.get('Issue')):
        bibtex += f"  issue = {{{paper_data.get('Issue')}}},\n"
    
    if pd.notna(paper_data.get('Article Number')):
        bibtex += f"  pages = {{{paper_data.get('Article Number')}}},\n"
    
    if pd.notna(paper_data.get('DOI')):
        bibtex += f"  doi = {{{paper_data.get('DOI')}}},\n"
    
    # Add abstract as note
    abstract = paper_data.get('Abstract', '')
    if pd.notna(abstract):
        abstract_short = str(abstract)[:200] + ("..." if len(str(abstract)) > 200 else "")
        bibtex += f"  abstract = {{{abstract_short}}},\n"
    
    # Remove trailing comma from last field
    bibtex = bibtex.rstrip(',\n') + "\n"
    bibtex += "}\n\n"
    
    return bibtex


def generate_bibtex_file(top_k_df, output_dir="data", k=100):
    """
    Generate BibTeX file for top k papers.
    
    Args:
        top_k_df: DataFrame containing top k papers
        output_dir: directory to save file
        k: number of papers (for naming)
    
    Returns:
        dict with file information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate BibTeX entries
    bibtex_content = ""
    for idx, row in top_k_df.iterrows():
        rank = row['Rank']
        bibtex_entry = create_bibtex_entry(rank, row.to_dict())
        bibtex_content += bibtex_entry
    
    # Save to file
    bibtex_file = os.path.join(output_dir, f"top_{k}_papers.bib")
    with open(bibtex_file, "w", encoding="utf-8") as f:
        f.write(bibtex_content)
    
    return {
        "file": bibtex_file,
        "entries": len(top_k_df),
        "file_size": os.path.getsize(bibtex_file)
    }


def print_top_k_summary(top_k_df, k=100, show_rows=10):
    """
    Print summary of top k papers.
    
    Args:
        top_k_df: DataFrame containing top k papers
        k: number of papers
        show_rows: number of rows to display
    """
    print(f"\nTop {k} papers summary:")
    print(f"Total rows: {len(top_k_df)}")
    print(f"Total columns: {len(top_k_df.columns)}")
    print(f"\nFirst {show_rows} papers (ranked by similarity score):")
    print("-" * 100)
    
    display_cols = ["Rank", "Similarity_Score", "Article Title", "Publication Year"]
    available_cols = [col for col in display_cols if col in top_k_df.columns]
    
    print(top_k_df[available_cols].head(show_rows).to_string(index=False))


def save_embeddings(paper_embeddings, interest_embedding, interest_text, 
                   output_dir="data/embeddings"):
    """
    Save embeddings to pickle files.
    
    Args:
        paper_embeddings: dictionary of paper embeddings
        interest_embedding: query embedding
        interest_text: query text
        output_dir: directory to save files
    
    Returns:
        dict with file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    paper_emb_file = os.path.join(output_dir, "paper_embeddings.pkl")
    interest_emb_file = os.path.join(output_dir, "interest_embedding.pkl")
    interest_text_file = os.path.join(output_dir, "interest_text.txt")
    
    with open(paper_emb_file, "wb") as f:
        pickle.dump(paper_embeddings, f)
    
    with open(interest_emb_file, "wb") as f:
        pickle.dump(interest_embedding, f)
    
    with open(interest_text_file, "w") as f:
        f.write(interest_text)
    
    return {
        "paper_embeddings": paper_emb_file,
        "interest_embedding": interest_emb_file,
        "interest_text": interest_text_file
    }


def load_embeddings(output_dir="data/embeddings"):
    """
    Load embeddings from pickle files.
    
    Args:
        output_dir: directory containing embedding files
    
    Returns:
        tuple of (paper_embeddings, interest_embedding, interest_text)
    """
    paper_emb_file = os.path.join(output_dir, "paper_embeddings.pkl")
    interest_emb_file = os.path.join(output_dir, "interest_embedding.pkl")
    interest_text_file = os.path.join(output_dir, "interest_text.txt")
    
    with open(paper_emb_file, "rb") as f:
        paper_embeddings = pickle.load(f)
    
    with open(interest_emb_file, "rb") as f:
        interest_embedding = pickle.load(f)
    
    with open(interest_text_file, "r") as f:
        interest_text = f.read()
    
    return paper_embeddings, interest_embedding, interest_text