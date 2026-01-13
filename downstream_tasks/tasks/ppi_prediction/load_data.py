"""
Data loading utilities for PPI prediction task.
Handles loading CSV data and applying PTM modifications to sequences.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


def apply_ptm_modification(sequence: str, ptm_type: str, site: int, aa: str) -> str:
    """
    Apply PTM modification to a sequence at the specified site.
    
    @param sequence: Original protein sequence (0-indexed)
    @param ptm_type: Type of PTM (e.g., 'Phos', 'Acety', 'Methyl')
    @param site: Site position (1-indexed, will be converted to 0-indexed)
    @param aa: Original amino acid at the site (for validation)
    @returns: Modified sequence with PTM token inserted
    """
    # Convert site from 1-indexed to 0-indexed
    site_idx = site - 1
    
    # Validate site index
    if site_idx < 0 or site_idx >= len(sequence):
        raise ValueError(f"Site {site} is out of range for sequence of length {len(sequence)}")
    
    # Validate amino acid matches
    if sequence[site_idx] != aa:
        raise ValueError(
            f"Amino acid mismatch at site {site}: expected {aa}, found {sequence[site_idx]}"
        )
    
    # Map PTM type to PTM token (based on PTMTokenizer in tokenizer.py and actual data analysis)
    # Data analysis shows:
    # - Phos: 86.13% (S: 55.26%, Y: 28.05%, T: 16.70%)
    # - Ac: 6.75% (K: 100%) - Acetylation
    # - Me: 3.56% (K: 73.05%, R: 26.95%) - Methylation
    # - Sumo: 1.77% (K: 100%) - SUMOylation (no direct token, use generic PTM)
    # - Ub: 1.68% (K: 100%) - Ubiquitination (no direct token, use generic PTM)
    # - Glyco: 0.11% (S: 80%, T: 20%) - Glycosylation
    # PTM tokens defined in tokenizer:
    # "PTM", "<N-linked (GlcNAc...) asparagine>", "<Pyrrolidone carboxylic acid>",
    # "<Phosphoserine>", "<Phosphothreonine>", "<N-acetylalanine>", "<N-acetylmethionine>",
    # "<N6-acetyllysine>", "<Phosphotyrosine>", "<S-diacylglycerol cysteine>",
    # "<N6-(pyridoxal phosphate)lysine>", "<N-acetylserine>", "<N6-carboxylysine>",
    # "<N6-succinyllysine>", "<S-palmitoyl cysteine>", "<O-(pantetheine 4'-phosphoryl)serine>",
    # "<Sulfotyrosine>", "<O-linked (GalNAc...) threonine>", "<Omega-N-methylarginine>",
    # "<N-myristoyl glycine>", "<4-hydroxyproline>", "<Asymmetric dimethylarginine>",
    # "<N5-methylglutamine>", "<4-aspartylphosphate>", "<S-geranylgeranyl cysteine>",
    # "<4-carboxyglutamate>"
    ptm_token_map = {
        # Phosphorylation (Phos) - 86.13%
        'Phos': {
            'S': '<Phosphoserine>',
            'T': '<Phosphothreonine>',
            'Y': '<Phosphotyrosine>'
        },
        # Acetylation (Ac) - 6.75%
        'Ac': {
            'K': '<N6-acetyllysine>',
            'A': '<N-acetylalanine>',
            'M': '<N-acetylmethionine>',
            'S': '<N-acetylserine>'
        },
        # Also support "Acety" for compatibility
        'Acety': {
            'K': '<N6-acetyllysine>',
            'A': '<N-acetylalanine>',
            'M': '<N-acetylmethionine>',
            'S': '<N-acetylserine>'
        },
        # Methylation (Me) - 3.56%
        'Me': {
            'R': '<Omega-N-methylarginine>',
            'K': 'PTM',  # K methylation not in tokenizer, use generic PTM
            'Q': '<N5-methylglutamine>'
        },
        # Also support "Methyl" for compatibility
        'Methyl': {
            'R': '<Omega-N-methylarginine>',
            'K': 'PTM',  # K methylation not in tokenizer, use generic PTM
            'Q': '<N5-methylglutamine>'
        },
        # SUMOylation (Sumo) - 1.77% - no direct token in tokenizer
        'Sumo': {
            'K': 'PTM'  # Use generic PTM token
        },
        'SUMO': {
            'K': 'PTM'  # Use generic PTM token
        },
        # Ubiquitination (Ub) - 1.68% - no direct token in tokenizer
        'Ub': {
            'K': 'PTM'  # Use generic PTM token
        },
        'Ubiquitin': {
            'K': 'PTM'  # Use generic PTM token
        },
        # Glycosylation (Glyco) - 0.11%
        'Glyco': {
            'S': 'PTM',  # O-linked glycosylation on S - use generic PTM or O-linked if T
            'T': '<O-linked (GalNAc...) threonine>',
            'N': '<N-linked (GlcNAc...) asparagine>'
        },
        'Glycosylation': {
            'S': 'PTM',
            'T': '<O-linked (GalNAc...) threonine>',
            'N': '<N-linked (GlcNAc...) asparagine>'
        },
        # Other PTM types (for completeness, based on tokenizer)
        'Succinyl': {
            'K': '<N6-succinyllysine>'
        },
        'Palmitoyl': {
            'C': '<S-palmitoyl cysteine>'
        },
        'Myristoyl': {
            'G': '<N-myristoyl glycine>'
        },
        'Geranylgeranyl': {
            'C': '<S-geranylgeranyl cysteine>'
        },
        'Diacylglycerol': {
            'C': '<S-diacylglycerol cysteine>'
        },
        'Sulfotyrosine': {
            'Y': '<Sulfotyrosine>'
        },
        'Hydroxyproline': {
            'P': '<4-hydroxyproline>'
        },
        'Carboxyglutamate': {
            'E': '<4-carboxyglutamate>'
        },
        'Aspartylphosphate': {
            'D': '<4-aspartylphosphate>'
        },
        'Dimethylarginine': {
            'R': '<Asymmetric dimethylarginine>'
        },
        'Pyrrolidone': {
            'Q': '<Pyrrolidone carboxylic acid>'
        },
        'Pantetheine': {
            'S': "<O-(pantetheine 4'-phosphoryl)serine>"
        },
        'Pyridoxal': {
            'K': '<N6-(pyridoxal phosphate)lysine>'
        },
        'Carboxylysine': {
            'K': '<N6-carboxylysine>'
        }
    }
    
    # Get PTM token
    ptm_token = None
    if ptm_type in ptm_token_map:
        if aa in ptm_token_map[ptm_type]:
            ptm_token = ptm_token_map[ptm_type][aa]
    
    # If no specific token found, use generic PTM token
    if ptm_token is None:
        # For unknown PTM types, we'll use a generic approach
        # Insert PTM token before the amino acid
        ptm_token = 'PTM'
        # For now, we'll just replace the amino acid with the PTM token
        # In practice, you might want to insert it before or use a different strategy
        modified_seq = sequence[:site_idx] + ptm_token + sequence[site_idx + 1:]
    else:
        # Replace the amino acid with the PTM token
        modified_seq = sequence[:site_idx] + ptm_token + sequence[site_idx + 1:]
    
    return modified_seq


def load_ppi_data(
    csv_path: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load PPI data from CSV file and split into train/valid/test sets.
    
    @param csv_path: Path to CSV file
    @param train_ratio: Ratio for training set (default: 0.7)
    @param valid_ratio: Ratio for validation set (default: 0.15)
    @param test_ratio: Ratio for test set (default: 0.15)
    @param random_seed: Random seed for reproducibility
    @returns: Tuple of (train_df, valid_df, test_df)
    """
    print(f"📖 Loading PPI data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = [
        'Uniprot_sequence', 'Int_uniprot_sequence', 'PTM', 'Site', 'AA', 'Effect'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Drop rows with NaN values in required columns
    df = df.dropna(subset=required_columns)
    
    # Filter out empty sequences
    df = df[df['Uniprot_sequence'].str.len() > 0]
    df = df[df['Int_uniprot_sequence'].str.len() > 0]
    
    # Convert Effect to binary label
    # Enhance/Induce (case-insensitive) = 1 (enhances interaction)
    # Inhibit (case-insensitive) = 0 (inhibits interaction)
    df['Effect_lower'] = df['Effect'].astype(str).str.strip().str.lower()
    df['label'] = df['Effect_lower'].apply(
        lambda x: 1 if x in ['enhance', 'induce'] else 0
    )
    df = df.drop(columns=['Effect_lower'])  # Clean up temporary column
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split data
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    train_df = df.iloc[:n_train].reset_index(drop=True)
    valid_df = df.iloc[n_train:n_train + n_valid].reset_index(drop=True)
    test_df = df.iloc[n_train + n_valid:].reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Training: {len(train_df)} samples")
    print(f"   Validation: {len(valid_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    return train_df, valid_df, test_df


def prepare_sequences_and_labels_for_embedding_generation(df: pd.DataFrame) -> Tuple[List[str], List[str], List[int]]:
    """
    为embedding generation准备序列和标签。
    只返回原始序列（binder和wt），不生成PTM序列。
    
    @param df: DataFrame with PPI data
    @returns: Tuple of (binder_sequences, wt_sequences, labels)
    """
    binder_sequences = []
    wt_sequences = []
    labels = []
    
    for _, row in df.iterrows():
        # Binder sequence (interaction partner)
        binder_seq = str(row['Int_uniprot_sequence']).strip()
        
        # Wild-type sequence (target protein)
        wt_seq = str(row['Uniprot_sequence']).strip()
        
        # Label (Enhance/Induce=1, Inhibit=0)
        label = int(row['label'])
        
        binder_sequences.append(binder_seq)
        wt_sequences.append(wt_seq)
        labels.append(label)
    
    return binder_sequences, wt_sequences, labels

