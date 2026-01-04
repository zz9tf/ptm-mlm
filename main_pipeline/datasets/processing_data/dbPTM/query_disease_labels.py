#!/usr/bin/env python3
"""
Query disease labels from UniProt for dbPTM sequences.

This script reads dbPTM data and queries UniProt API to determine
if each protein sequence is disease-associated.
"""

import requests
import pandas as pd
from io import StringIO
from typing import List, Optional
import time
from pathlib import Path


UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"


def chunk(lst: List[str], n: int = 100, max_query_length: int = 8000) -> List[List[str]]:
    """
    Split a list into chunks, ensuring query string doesn't exceed max length.
    
    Args:
        lst: List to chunk
        n: Initial chunk size
        max_query_length: Maximum query string length (UniProt has limits)
        
    Yields:
        Chunks of the list
    """
    i = 0
    while i < len(lst):
        chunk_size = n
        # Build query to check length
        test_query = " OR ".join([f"(accession:{a})" for a in lst[i:i+chunk_size]])
        
        # If query is too long, reduce chunk size
        while len(test_query) > max_query_length and chunk_size > 1:
            chunk_size = max(1, chunk_size // 2)
            test_query = " OR ".join([f"(accession:{a})" for a in lst[i:i+chunk_size]])
        
        yield lst[i:i+chunk_size]
        i += chunk_size


def uniprot_disease_labels(accessions: List[str]) -> pd.DataFrame:
    """
    Query UniProt for disease annotations.
    
    Returns a DataFrame with columns:
    - accession
    - disease_text (may be empty)
    - disease_label (1 if any disease annotation exists else 0)
    
    Args:
        accessions: List of UniProt accession IDs
        
    Returns:
        DataFrame with disease labels
    """
    rows = []
    # Use smaller batch size to avoid query string length limits
    batch_size = 100
    total_batches = (len(accessions) + batch_size - 1) // batch_size
    
    print(f"📡 Querying UniProt for {len(accessions)} accessions in ~{total_batches} batches (batch size: {batch_size})...")
    
    for batch_idx, batch in enumerate(chunk(accessions, batch_size), 1):
        # query: accession:P04637 OR accession:Q9Y6K9 ...
        q = " OR ".join([f"(accession:{a})" for a in batch])

        params = {
            "query": q,
            "format": "tsv",
            "fields": "accession,protein_name,gene_primary,cc_disease"
        }
        
        try:
            r = requests.get(UNIPROT_STREAM, params=params, timeout=60)
            r.raise_for_status()

            df = pd.read_csv(StringIO(r.text), sep="\t")
            # UniProt returns column name like "Involvement in disease" for cc_disease
            # Use case-insensitive matching to find the disease column
            disease_col = [c for c in df.columns if "disease" in c.lower()]
            if disease_col:
                df["disease_text"] = df[disease_col[0]].fillna("").astype(str)
            else:
                df["disease_text"] = ""
            # disease_label = 1 if disease_text is not empty, 0 otherwise
            df["disease_label"] = (df["disease_text"].str.len() > 0).astype(int)

            rows.append(df[["Entry", "Protein names", "Gene Names (primary)", "disease_text", "disease_label"]]
                        .rename(columns={
                            "Entry": "accession",
                            "Protein names": "protein_name",
                            "Gene Names (primary)": "gene"
                        }))
            
            print(f"  ✓ Batch {batch_idx}/{total_batches}: {len(df)} results")
            
        except Exception as e:
            print(f"  ⚠️  Batch {batch_idx} failed: {e}")
            # If query string is too long, try smaller batches
            if "400" in str(e) and len(batch) > 10:
                print(f"     Query string too long, trying smaller batch...")
                # Split into smaller chunks and retry
                mid = len(batch) // 2
                for sub_batch in [batch[:mid], batch[mid:]]:
                    try:
                        q_sub = " OR ".join([f"(accession:{a})" for a in sub_batch])
                        params_sub = {
                            "query": q_sub,
                            "format": "tsv",
                            "fields": "accession,protein_name,gene_primary,cc_disease"
                        }
                        r_sub = requests.get(UNIPROT_STREAM, params=params_sub, timeout=60)
                        r_sub.raise_for_status()
                        df_sub = pd.read_csv(StringIO(r_sub.text), sep="\t")
                        disease_col = [c for c in df_sub.columns if "disease" in c.lower()]
                        if disease_col:
                            df_sub["disease_text"] = df_sub[disease_col[0]].fillna("").astype(str)
                        else:
                            df_sub["disease_text"] = ""
                        df_sub["disease_label"] = (df_sub["disease_text"].str.len() > 0).astype(int)
                        rows.append(df_sub[["Entry", "Protein names", "Gene Names (primary)", "disease_text", "disease_label"]]
                                    .rename(columns={
                                        "Entry": "accession",
                                        "Protein names": "protein_name",
                                        "Gene Names (primary)": "gene"
                                    }))
                        time.sleep(0.1)
                    except Exception as e2:
                        print(f"     Sub-batch also failed: {e2}")
                        # Add empty rows for failed sub-batch
                        rows.append(pd.DataFrame({
                            "accession": sub_batch,
                            "protein_name": [None] * len(sub_batch),
                            "gene": [None] * len(sub_batch),
                            "disease_text": [""] * len(sub_batch),
                            "disease_label": [0] * len(sub_batch),
                        }))
            else:
                # Add empty rows for failed batch
                rows.append(pd.DataFrame({
                    "accession": batch,
                    "protein_name": [None] * len(batch),
                    "gene": [None] * len(batch),
                    "disease_text": [""] * len(batch),
                    "disease_label": [0] * len(batch),
                }))
        
        # Rate limiting
        time.sleep(0.1)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["accession", "protein_name", "gene", "disease_text", "disease_label"]
    )
    
    # 保证输出包含所有输入（有些 accession 可能查不到）
    out = out.drop_duplicates(subset=["accession"])
    missing = sorted(set(accessions) - set(out["accession"]))
    if missing:
        out = pd.concat([out, pd.DataFrame({
            "accession": missing,
            "protein_name": [None]*len(missing),
            "gene": [None]*len(missing),
            "disease_text": [""]*len(missing),
            "disease_label": [0]*len(missing),
        })], ignore_index=True)
    
    return out


def query_disease_labels_for_dbptm(
    input_file: str = "dbptm_data.csv",
    output_file: str = None,
    accession_col: str = "AC_ID"
) -> None:
    """
    Query disease labels and add them to the dataset, saving to a new file.
    
    Args:
        input_file: Path to dataset CSV file
        output_file: Path to save updated dataset (if None, creates new filename)
        accession_col: Column name containing UniProt accession IDs
    """
    print("🚀 Starting disease label query...\n")
    
    # Read dataset
    print(f"📂 Reading dataset from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   Loaded {len(df):,} records")
    
    # Get unique accessions
    accessions = df[accession_col].dropna().unique().tolist()
    print(f"📊 Found {len(accessions)} unique UniProt accessions\n")
    
    # Query disease labels
    disease_df = uniprot_disease_labels(accessions)
    
    # Debug: Check query results
    print(f"\n📊 Query results summary:")
    print(f"   Total accessions queried: {len(accessions)}")
    print(f"   Results returned: {len(disease_df)}")
    print(f"   Disease-associated in results: {disease_df['disease_label'].sum()} ({disease_df['disease_label'].sum()/len(disease_df)*100:.2f}%)")
    
    # Merge labels into dataset
    print(f"\n🔗 Merging disease labels into dataset...")
    
    # Merge on accession column
    df = df.merge(
        disease_df[["accession", "disease_text", "disease_label"]],
        left_on=accession_col,
        right_on="accession",
        how="left",
        suffixes=("", "_disease")
    )
    
    # Remove duplicate accession column
    if "accession" in df.columns and accession_col in df.columns:
        df = df.drop(columns=["accession"])
    
    # Fill NaN values for disease_label (accessions not found in UniProt)
    # Check how many matched
    matched_count = df["disease_label"].notna().sum()
    print(f"   Matched records: {matched_count:,} / {len(df):,} ({matched_count/len(df)*100:.2f}%)")
    
    df["disease_label"] = df["disease_label"].fillna(0).astype(int)
    df["disease_text"] = df["disease_text"].fillna("")
    
    # Determine output file (create new filename if not specified)
    # Handle empty string as None (can happen when called from shell script)
    if output_file is None or (isinstance(output_file, str) and output_file.strip() == ""):
        # Create new filename: input_file with "_with_disease_labels" suffix
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_with_disease_labels{input_path.suffix}")
    
    # Save updated dataset to new file
    print(f"\n💾 Saving updated dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Print summary
    disease_count = df["disease_label"].sum()
    print(f"\n✅ Query complete!")
    print(f"   Total records: {len(df):,}")
    print(f"   Disease-associated: {disease_count:,} ({disease_count/len(df)*100:.2f}%)")
    print(f"   New dataset saved to: {output_file}")


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "dbptm_data.csv"
    # Handle empty string as None (can happen when called from shell script)
    output_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
    accession_col = sys.argv[3] if len(sys.argv) > 3 else "AC_ID"
    
    query_disease_labels_for_dbptm(input_file, output_file, accession_col)
