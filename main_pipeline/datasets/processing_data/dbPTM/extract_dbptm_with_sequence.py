#!/usr/bin/env python3
"""
Extract dbPTM data with original sequences.

This script:
1. Extracts dbPTM records from PTM_added_gene_symbol.csv (SOURCE='dbPTM')
2. Matches with sequence data from combined.csv using UNIPROT_ID
3. Removes unnecessary columns: FUNCTIONAL_ROLE, SOURCE, FUNCTIONAL_ROLE_STANDARDIZED, query, MODIFICATION_TYPE
4. Saves to processing_data/dbptm_data.csv
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import sys


def extract_dbptm_with_sequence(
    archive_file: str = "/home/zz/zheng/PTM/Post_Translational_Modification/Archive/PTM_added_gene_symbol.csv",
    combined_file: str = "/home/zz/zheng/ptm-mlm/main_pipeline/datasets/combined.csv",
    output_file: str = "/home/zz/zheng/ptm-mlm/main_pipeline/datasets/processing_data/dbptm_data.csv",
    source_value: str = "dbPTM",
    chunk_size: int = 100000
) -> Optional[str]:
    """
    Extract dbPTM data with sequences.
    
    Logic:
    - Filter dbPTM records from PTM_added_gene_symbol.csv
    - Match with combined.csv using UNIPROT_ID = AC_ID
    - Keep only necessary columns
    
    Args:
        archive_file: Path to PTM_added_gene_symbol.csv
        combined_file: Path to combined.csv (contains sequences)
        output_file: Path to save extracted data
        source_value: Value to filter in SOURCE column
        chunk_size: Chunk size for reading large files
        
    Returns:
        Path to output file if successful, None otherwise
    """
    print("🚀 Starting dbPTM data extraction with sequences...\n")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Columns to keep from PTM_added_gene_symbol.csv
    # Remove: FUNCTIONAL_ROLE, SOURCE, FUNCTIONAL_ROLE_STANDARDIZED, query, MODIFICATION_TYPE
    keep_cols_ptm = [
        'UNIPROT_ID',
        'LOCATION',
        'SPECIES',
        'PMID',
        'symbol'
    ]
    
    # Columns from combined.csv
    keep_cols_seq = [
        'AC_ID',
        'ori_seq',
        'ptm_seq',
        'num_ptm_sites'
    ]
    
    print(f"📂 Step 1: Getting dbPTM UNIPROT_IDs from Archive...")
    print(f"   File: {archive_file}")
    print(f"   Filtering for SOURCE='{source_value}'")
    
    # Get set of UNIPROT_IDs that have dbPTM records
    dbptm_uniprot_ids = set()
    total_dbptm_records = 0
    
    try:
        for chunk in pd.read_csv(archive_file, chunksize=chunk_size, usecols=['UNIPROT_ID', 'SOURCE'], low_memory=False):
            # Filter for dbPTM
            dbptm_chunk = chunk[chunk['SOURCE'] == source_value]
            total_dbptm_records += len(dbptm_chunk)
            
            if len(dbptm_chunk) > 0:
                dbptm_uniprot_ids.update(dbptm_chunk['UNIPROT_ID'].dropna().unique())
        
        print(f"   ✓ Found {len(dbptm_uniprot_ids):,} unique UNIPROT_IDs with SOURCE='{source_value}'")
        print(f"   Total dbPTM records: {total_dbptm_records:,}")
        
        if len(dbptm_uniprot_ids) == 0:
            print(f"   ⚠️  No dbPTM records found!")
            return None
        
    except Exception as e:
        print(f"   ❌ Error reading dbPTM data: {e}")
        return None
    
    print(f"\n📂 Step 2: Filtering combined.csv for dbPTM sequences...")
    print(f"   File: {combined_file}")
    print(f"   Only keeping AC_IDs that exist in dbPTM data (intersection)")
    
    # Load and filter combined.csv
    try:
        # First, read in chunks to filter
        filtered_chunks = []
        total_sequences = 0
        filtered_sequences = 0
        
        for chunk in pd.read_csv(combined_file, chunksize=chunk_size, low_memory=False):
            total_sequences += len(chunk)
            # Filter for AC_IDs that are in dbPTM
            filtered_chunk = chunk[chunk['AC_ID'].isin(dbptm_uniprot_ids)].copy()
            if len(filtered_chunk) > 0:
                filtered_chunks.append(filtered_chunk)
                filtered_sequences += len(filtered_chunk)
        
        if not filtered_chunks:
            print(f"   ⚠️  No sequences found in intersection!")
            return None
        
        # Concatenate filtered chunks
        filtered_df = pd.concat(filtered_chunks, ignore_index=True)
        
        print(f"   ✓ Total sequences in combined.csv: {total_sequences:,}")
        print(f"   ✓ Sequences in intersection (dbPTM): {filtered_sequences:,} ({filtered_sequences/total_sequences*100:.2f}%)")
        print(f"   ✓ Unique AC_IDs in filtered data: {filtered_df['AC_ID'].nunique():,}")
        
    except Exception as e:
        print(f"   ❌ Error filtering combined.csv: {e}")
        return None
    
    # The filtered_df is our final result (from combined.csv, filtered by dbPTM)
    merged_df = filtered_df
    
    # Keep only columns from combined.csv (no need to merge with PTM_added_gene_symbol data)
    # The filtered data already contains all sequence information
    
    print(f"\n💾 Step 4: Saving to {output_file}...")
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"   ✓ Saved {len(merged_df):,} records")
        print(f"   Columns: {', '.join(merged_df.columns.tolist())}")
    except Exception as e:
        print(f"   ❌ Error saving file: {e}")
        return None
    
    # Print summary
    print(f"\n✅ Extraction complete!")
    print(f"   Total records (from combined.csv, filtered by dbPTM): {len(merged_df):,}")
    print(f"   Unique AC_IDs: {merged_df['AC_ID'].nunique():,}")
    print(f"   All records are from combined.csv and have dbPTM source")
    print(f"   Output file: {output_file}")
    
    return output_file


if __name__ == "__main__":
    archive_file = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/zz/zheng/PTM/Post_Translational_Modification/Archive/PTM_added_gene_symbol.csv"
    combined_file = sys.argv[2] if len(sys.argv) > 2 else \
        "/home/zz/zheng/ptm-mlm/main_pipeline/datasets/combined.csv"
    output_file = sys.argv[3] if len(sys.argv) > 3 else \
        "/home/zz/zheng/ptm-mlm/main_pipeline/datasets/processing_data/dbptm_data.csv"
    
    result = extract_dbptm_with_sequence(archive_file, combined_file, output_file)
    
    if result:
        print(f"\n✨ Success! Data saved to: {result}")
    else:
        print("\n❌ Extraction failed. Please check the error messages above.")
        sys.exit(1)