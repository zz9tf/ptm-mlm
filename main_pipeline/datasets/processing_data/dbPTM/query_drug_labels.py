#!/usr/bin/env python3
"""
Query drug target labels from ChEMBL for dbPTM sequences.

This script reads dbPTM data and queries ChEMBL API to determine
if each protein sequence is a drug target.
"""

import time
import requests
import pandas as pd
from typing import List, Optional
from pathlib import Path
import sys


def chembl_drug_target_labels(accessions: List[str], sleep: float = 0.05) -> pd.DataFrame:
    """
    Query ChEMBL for drug target annotations.
    
    drug_target_label:
      1 if UniProt accession appears in any ChEMBL drug mechanism record (proxy for drug target)
      0 otherwise
    
    Args:
        accessions: List of UniProt accession IDs
        sleep: Sleep time between requests (seconds)
        
    Returns:
        DataFrame with drug target labels
    """
    base = "https://www.ebi.ac.uk/chembl/api/data"
    out = []
    total = len(accessions)
    
    print(f"📡 Querying ChEMBL for {total} accessions...")
    print("   (This may take a while due to rate limiting)\n")
    
    for idx, acc in enumerate(accessions, 1):
        try:
            # step1: target search by UniProt accession (target_components)
            url1 = f"{base}/target.json"
            params1 = {"target_components__accession": acc, "limit": 1000}
            r1 = requests.get(url1, params=params1, timeout=60)
            r1.raise_for_status()
            data1 = r1.json()
            targets = [t["target_chembl_id"] for t in data1.get("targets", [])]

            label = 0
            hit_target = None

            # step2: check mechanism records for any target_chembl_id
            for tid in targets:
                url2 = f"{base}/mechanism.json"
                params2 = {"target_chembl_id": tid, "limit": 1}  # any record is enough
                r2 = requests.get(url2, params=params2, timeout=60)
                r2.raise_for_status()
                data2 = r2.json()
                if data2.get("mechanisms"):
                    label = 1
                    hit_target = tid
                    break
                time.sleep(sleep)

            out.append({
                "accession": acc,
                "chembl_target_hit": hit_target,
                "drug_target_label": label
            })
            
            if idx % 100 == 0:
                drug_count = sum(r["drug_target_label"] for r in out)
                print(f"  ✓ Progress: {idx}/{total} ({idx/total*100:.1f}%) - "
                      f"Drug targets found: {drug_count}")
            
        except Exception as e:
            print(f"  ⚠️  Error querying {acc}: {e}")
            out.append({
                "accession": acc,
                "chembl_target_hit": None,
                "drug_target_label": 0
            })
        
        time.sleep(sleep)

    return pd.DataFrame(out)


def query_drug_labels_for_dbptm(
    input_file: str = "dbptm_data.csv",
    output_file: str = None,
    accession_col: str = "AC_ID",
    sleep: float = 0.05
) -> None:
    """
    Query drug target labels and add them to the dataset, saving to a new file.
    
    Args:
        input_file: Path to dataset CSV file
        output_file: Path to save updated dataset (if None, creates new filename)
        accession_col: Column name containing UniProt accession IDs
        sleep: Sleep time between requests (seconds)
    """
    print("🚀 Starting drug target label query...\n")
    
    # Read dataset
    print(f"📂 Reading dataset from: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   Loaded {len(df):,} records")
    
    # Get unique accessions
    accessions = df[accession_col].dropna().unique().tolist()
    print(f"📊 Found {len(accessions)} unique UniProt accessions\n")
    
    # Query drug target labels
    drug_df = chembl_drug_target_labels(accessions, sleep=sleep)
    
    # Merge labels into dataset
    print(f"\n🔗 Merging drug target labels into dataset...")
    
    # Merge on accession column
    df = df.merge(
        drug_df[["accession", "chembl_target_hit", "drug_target_label"]],
        left_on=accession_col,
        right_on="accession",
        how="left",
        suffixes=("", "_drug")
    )
    
    # Remove duplicate accession column
    if "accession" in df.columns and accession_col in df.columns:
        df = df.drop(columns=["accession"])
    
    # Fill NaN values for drug_target_label (accessions not found in ChEMBL)
    df["drug_target_label"] = df["drug_target_label"].fillna(0).astype(int)
    df["chembl_target_hit"] = df["chembl_target_hit"].fillna("")
    
    # Determine output file (create new filename if not specified)
    # Handle empty string as None (can happen when called from shell script)
    if output_file is None or (isinstance(output_file, str) and output_file.strip() == ""):
        # Create new filename: input_file with "_with_drug_labels" suffix
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_with_drug_labels{input_path.suffix}")
    
    # Save updated dataset to new file
    print(f"\n💾 Saving updated dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Print summary
    drug_count = df["drug_target_label"].sum()
    print(f"\n✅ Query complete!")
    print(f"   Total records: {len(df):,}")
    print(f"   Drug targets: {drug_count:,} ({drug_count/len(df)*100:.2f}%)")
    print(f"   New dataset saved to: {output_file}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "dbptm_data.csv"
    # Handle empty string as None (can happen when called from shell script)
    output_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
    accession_col = sys.argv[3] if len(sys.argv) > 3 else "AC_ID"
    sleep = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    
    query_drug_labels_for_dbptm(input_file, output_file, accession_col, sleep)
