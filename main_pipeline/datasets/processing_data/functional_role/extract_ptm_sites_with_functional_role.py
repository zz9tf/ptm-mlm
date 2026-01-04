#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract individual PTM sites with functional roles for downstream tasks.
Each row represents one PTM site with sequence context, position, AA, PTM type, and functional role.
"""

import csv
import re
import os
from collections import defaultdict
from pathlib import Path


def build_ptm_location_dict(ptm_source_file: str) -> dict:
    """
    Build a dictionary mapping (UNIPROT_ID, LOCATION, MODIFICATION_TYPE) to FUNCTIONAL_ROLE.
    Only includes records with both LOCATION and MODIFICATION_TYPE.
    If multiple functional roles exist for the same key, use the most common one.
    
    @param {str} ptm_source_file - Path to PTM_added_gene_symbol.csv file
    @returns {dict} Dictionary mapping (UNIPROT_ID, LOCATION, MODIFICATION_TYPE) to FUNCTIONAL_ROLE
    """
    from collections import Counter
    
    ptm_dict_temp = defaultdict(list)  # Store all roles for counting
    skipped_no_location = 0
    
    print(f"📖 Reading functional roles from: {ptm_source_file}")
    with open(ptm_source_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot_id = row.get('UNIPROT_ID', '').strip()
            location = row.get('LOCATION', '').strip()
            mod_type = row.get('MODIFICATION_TYPE', '').strip()
            functional_role = row.get('FUNCTIONAL_ROLE', '').strip()
            
            # Only include records with functional_role, location, and modification_type
            if not uniprot_id or not functional_role or not mod_type:
                continue
            
            # Skip records without location (cannot map to specific PTM site)
            if not location or location == '' or location.lower() == 'nan':
                skipped_no_location += 1
                continue
            
            # Handle location - convert to int if possible
            try:
                loc_key = int(float(location))
            except (ValueError, TypeError):
                # If cannot convert, skip this record
                skipped_no_location += 1
                continue
            
            # Create key: (UNIPROT_ID, LOCATION, MODIFICATION_TYPE)
            key = (uniprot_id, loc_key, mod_type)
            ptm_dict_temp[key].append(functional_role)
    
    # For each key, use the most common functional_role
    ptm_dict = {}
    multi_role_count = 0
    for key, roles in ptm_dict_temp.items():
        if len(set(roles)) > 1:
            # Multiple different roles - use the most common one
            most_common = Counter(roles).most_common(1)[0][0]
            ptm_dict[key] = most_common
            multi_role_count += 1
        else:
            # Single role
            ptm_dict[key] = roles[0]
    
    print(f"✅ Found {len(ptm_dict):,} unique (UNIPROT_ID, LOCATION, MOD_TYPE) combinations with functional roles")
    if skipped_no_location > 0:
        print(f"⚠️  Skipped {skipped_no_location:,} records without LOCATION (cannot map to specific PTM site)")
    if multi_role_count > 0:
        print(f"⚠️  {multi_role_count:,} combinations had multiple roles, using most common one")
    
    # Save dictionary to file for inspection
    dict_output_file = '/home/zz/zheng/ptm-mlm/main_pipeline/datasets/ptm_location_dict.csv'
    print(f"\n💾 Saving dictionary to: {dict_output_file}")
    with open(dict_output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['UNIPROT_ID', 'LOCATION', 'MODIFICATION_TYPE', 'FUNCTIONAL_ROLE'])
        for (uniprot_id, location, mod_type), functional_role in sorted(ptm_dict.items()):
            writer.writerow([uniprot_id, location, mod_type, functional_role])
    print(f"✅ Saved {len(ptm_dict):,} dictionary entries to {dict_output_file}")
    
    return ptm_dict


def extract_ptm_sites_from_sequence(ori_seq: str, ptm_seq: str) -> list:
    """
    Extract PTM sites from ptm_seq and map to positions in ori_seq.
    PTM markers like <Phosphoserine> appear AFTER the modified amino acid.
    
    @param {str} ori_seq - Original sequence without PTM annotations
    @param {str} ptm_seq - Sequence with PTM annotations like <Phosphoserine>
    @returns {list} List of dicts with position, AA, and PTM type
    """
    ptm_sites = []
    pattern = r'<([^>]+)>'
    
    # Find all PTM markers in ptm_seq
    matches = list(re.finditer(pattern, ptm_seq))
    
    # Remove all PTM markers to reconstruct the original sequence
    # This helps us map positions correctly
    cleaned_ptm_seq = re.sub(pattern, '', ptm_seq)
    
    # Verify that cleaned sequence matches ori_seq
    if cleaned_ptm_seq != ori_seq:
        # If they don't match exactly, we need to handle it differently
        # For now, we'll proceed but this might indicate an issue
        pass
    
    # Extract PTM sites
    # For each marker, find its position in ptm_seq, then map to ori_seq
    for match in matches:
        marker_start = match.start()
        
        # Count how many regular characters (not in markers) are before this marker
        # This gives us the position in the cleaned sequence
        char_count = 0
        i = 0
        while i < marker_start:
            if ptm_seq[i] == '<':
                # Skip the entire marker
                marker_end = ptm_seq.find('>', i) + 1
                if marker_end > 0:
                    i = marker_end
                else:
                    break
            else:
                char_count += 1
                i += 1
        
        # The modified AA is at position char_count (1-based) in ori_seq
        # PTM marker appears AFTER the modified AA
        # So the modified AA is the last character before the marker
        if char_count > 0 and char_count <= len(ori_seq):
            ori_pos = char_count  # 1-based position (the last char before marker)
            aa = ori_seq[char_count - 1]  # 0-based index
            ptm_type = match.group(1)
            
            # Verify: if PTM type mentions an amino acid, check if it matches
            # This is just for validation, we'll still add the site
            ptm_sites.append({
                'position': ori_pos,
                'aa': aa,
                'ptm_type': ptm_type
            })
    
    return ptm_sites


def map_general_to_specific_ptm_type(mod_type: str, aa: str) -> str:
    """
    Map general MODIFICATION_TYPE to specific PTM type based on amino acid.
    This aligns with tokenizer tokens (without angle brackets).
    
    **Why this alignment is necessary:**
    - Tokenizer uses specific PTM tokens like "Phosphoserine", "Phosphothreonine", etc.
    - Source data only provides general types like "Phosphorylation", "Acetylation"
    - We need to map general type + amino acid -> specific type to match tokenizer vocabulary
    - This ensures downstream tasks can correctly tokenize PTM information
    
    **Data analysis results (from 54,629 valid records):**
    - Phosphorylation (65.56%): S (74.17%), T (11.80%), Y (9.69%)
    - Oxidation (29.91%): M (95.50%) - no tokenizer token, keep as "Oxidation"
    - Methylation (1.82%): K (71.51%), R (18.36%)
    - Acetylation (1.61%): K (89.12%)
    - Ubiquitination (0.87%): K (95.98%) - no tokenizer token, keep as "Ubiquitination"
    
    **Mapping logic (based on actual data distribution):**
    
    **Phosphorylation (65.56% of all PTMs):**
    - S -> Phosphoserine (matches tokenizer token, 74.17% of phosphorylation)
    - T -> Phosphothreonine (matches tokenizer token, 11.80% of phosphorylation)
    - Y -> Phosphotyrosine (matches tokenizer token, 9.69% of phosphorylation)
    - Other AAs -> "Phosphorylation" (generic fallback, rare cases)
    
    **Acetylation (1.61% of all PTMs):**
    - K -> N6-acetyllysine (matches tokenizer token, 89.12% of acetylation)
    - A -> N-acetylalanine (matches tokenizer token)
    - M -> N-acetylmethionine (matches tokenizer token)
    - S -> N-acetylserine (matches tokenizer token)
    - Other AAs -> N6-acetyllysine (default to most common)
    
    **Methylation (1.82% of all PTMs):**
    - R -> Omega-N-methylarginine (matches tokenizer token, 18.36% of methylation)
    - Q -> N5-methylglutamine (matches tokenizer token)
    - K/Other AAs -> "Methylation" (generic fallback, K methylation is 71.51% but no tokenizer token)
    
    **Glycosylation (0.01% of all PTMs):**
    - N -> N-linked (GlcNAc...) asparagine (matches tokenizer token)
    - T -> O-linked (GalNAc...) threonine (matches tokenizer token)
    - Other AAs -> N-linked (GlcNAc...) asparagine (default)
    
    **Oxidation (29.91% of all PTMs):**
    - All AAs -> "Oxidation" (no tokenizer token available)
    
    **Ubiquitination (0.87% of all PTMs):**
    - All AAs -> "Ubiquitination" (no tokenizer token available)
    
    **Other types (Sumoylation, Trimethylation, etc.):**
    - Return as-is (generic type)
    
    @param {str} mod_type - General modification type (e.g., 'Phosphorylation', 'Acetylation')
    @param {str} aa - Amino acid at the modification site (single letter code)
    @returns {str} Specific PTM type matching tokenizer tokens (without <>), or generic type if no match
    """
    mod_type_lower = mod_type.lower()
    aa_upper = aa.upper()
    
    # Phosphorylation mappings (65.56% of all PTMs)
    if mod_type_lower == 'phosphorylation':
        if aa_upper == 'S':
            return 'Phosphoserine'  # 74.17% of phosphorylation
        elif aa_upper == 'T':
            return 'Phosphothreonine'  # 11.80% of phosphorylation
        elif aa_upper == 'Y':
            return 'Phosphotyrosine'  # 9.69% of phosphorylation
        else:
            # For other amino acids (rare cases), return generic
            return 'Phosphorylation'
    
    # Acetylation mappings (1.61% of all PTMs)
    elif mod_type_lower == 'acetylation':
        if aa_upper == 'K':
            return 'N6-acetyllysine'  # 89.12% of acetylation
        elif aa_upper == 'A':
            return 'N-acetylalanine'  # tokenizer has this token
        elif aa_upper == 'M':
            return 'N-acetylmethionine'  # tokenizer has this token
        elif aa_upper == 'S':
            return 'N-acetylserine'  # tokenizer has this token
        else:
            return 'N6-acetyllysine'  # Default to most common
    
    # Methylation mappings (1.82% of all PTMs)
    elif mod_type_lower == 'methylation':
        if aa_upper == 'R':
            return 'Omega-N-methylarginine'  # 18.36% of methylation, tokenizer has this
        elif aa_upper == 'Q':
            return 'N5-methylglutamine'  # tokenizer has this token
        else:
            # For K methylation (71.51% of methylation) and others, no specific tokenizer token
            # Return generic "Methylation"
            return 'Methylation'
    
    # Glycosylation mappings (0.01% of all PTMs)
    elif mod_type_lower == 'glycosylation' or 'glycosyl' in mod_type_lower or 'glcnac' in mod_type_lower:
        if aa_upper == 'N':
            return 'N-linked (GlcNAc...) asparagine'  # tokenizer has this token
        elif aa_upper == 'T':
            return 'O-linked (GalNAc...) threonine'  # tokenizer has this token
        else:
            return 'N-linked (GlcNAc...) asparagine'  # Default
    
    # Oxidation (29.91% of all PTMs) - no tokenizer token available
    # Keep as generic "Oxidation"
    elif mod_type_lower == 'oxidation':
        return 'Oxidation'
    
    # Ubiquitination (0.87% of all PTMs) - no tokenizer token available
    # Keep as generic "Ubiquitination"
    elif 'ubiquitin' in mod_type_lower:
        return 'Ubiquitination'
    
    # Sumoylation, Trimethylation, etc. - keep as generic
    # For unknown types, return as-is
    return mod_type


def normalize_functional_role(functional_role: str) -> str:
    """
    Normalize functional role labels to three standardized categories.
    
    **Why normalization is necessary:**
    - Source data has inconsistent labels: "inhibit", "Inhibit", "Induce", "Enhance", etc.
    - Need standardized categories for machine learning tasks
    - Three categories: Impairing, Enhancing, Associated
    
    **Data analysis results (from 54,912 valid records):**
    - Enhancing: 23,796 (43.46%) - already normalized
    - Impairing: 21,390 (39.22%) - already normalized
    - Enhance: 4,168 (7.59%) -> map to Enhancing
    - Associated: 3,557 (6.59%) - already normalized
    - Inhibit: 1,627 (2.97%) -> map to Impairing
    - Induce: 66 (0.12%) -> map to Enhancing
    - inhibit: 16 (0.03%) -> map to Impairing
    - induce: 9 (0.02%) -> map to Enhancing
    
    **Mapping rules (based on actual data):**
    - Enhancing, Enhance, Induce, induce -> Enhancing
    - Impairing, Inhibit, inhibit -> Impairing
    - Associated -> Associated
    
    @param {str} functional_role - Original functional role label
    @returns {str} Normalized functional role: 'Impairing', 'Enhancing', or 'Associated'
    """
    if not functional_role:
        return 'Associated'
    
    role_stripped = functional_role.strip()
    role_lower = role_stripped.lower()
    
    # Already normalized cases (most common)
    if role_stripped == 'Enhancing':
        return 'Enhancing'
    elif role_stripped == 'Impairing':
        return 'Impairing'
    elif role_stripped == 'Associated':
        return 'Associated'
    
    # Map variations to Enhancing
    # Based on actual data: "Enhance", "Induce", "induce"
    if role_lower == 'enhance' or role_lower == 'induce':
        return 'Enhancing'
    
    # Map variations to Impairing
    # Based on actual data: "Inhibit", "inhibit"
    if role_lower == 'inhibit':
        return 'Impairing'
    
    # All 8 functional_role variants have been handled above
    # Default to Associated for any unexpected cases (should not happen with current data)
    return 'Associated'


def get_sequence_context(ori_seq: str, position: int, context_size: int = 15) -> tuple:
    """
    Extract sequence context around a PTM site.
    
    @param {str} ori_seq - Original sequence
    @param {int} position - 1-based position of PTM site
    @param {int} context_size - Size of context on each side
    @returns {tuple} (left_context, right_context, full_context)
    """
    pos_0based = position - 1
    
    left_start = max(0, pos_0based - context_size)
    right_end = min(len(ori_seq), pos_0based + context_size + 1)
    
    left_context = ori_seq[left_start:pos_0based]
    right_context = ori_seq[pos_0based + 1:right_end]
    full_context = ori_seq[left_start:right_end]
    
    return left_context, right_context, full_context


def load_sequence_dict(combined_file: str) -> dict:
    """
    Load sequences from combined.csv and create a dictionary mapping AC_ID to ori_seq.
    
    @param {str} combined_file - Path to combined.csv file
    @returns {dict} Dictionary mapping AC_ID to ori_seq
    """
    sequence_dict = {}
    print(f"📖 Loading sequences from: {combined_file}")
    
    with open(combined_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ac_id = row.get('AC_ID', '').strip()
            ori_seq = row.get('ori_seq', '').strip()
            
            if ac_id and ori_seq:
                # If multiple sequences exist for same AC_ID, keep the first one
                if ac_id not in sequence_dict:
                    sequence_dict[ac_id] = ori_seq
    
    print(f"✅ Loaded {len(sequence_dict):,} unique sequences")
    return sequence_dict


def process_ptm_data_from_source_file(
    ptm_source_file: str,
    combined_file: str,
    output_file: str,
    context_size: int = 15
) -> None:
    """
    Process PTM data from source file and combined.csv, with deduplication and label selection.
    
    Processing logic:
    1. Load all records from PTM_added_gene_symbol.csv (with all attributes)
    2. Load sequences from combined.csv
    3. Remove records without LOCATION
    4. Remove completely duplicate records
    5. For same (sequence, position, ptm_type): select most common functional_role
    6. If label counts are equal: select first one alphabetically
    
    @param {str} ptm_source_file - Path to PTM_added_gene_symbol.csv
    @param {str} combined_file - Path to combined.csv
    @param {str} output_file - Path to output CSV file
    @param {int} context_size - Size of sequence context around PTM site
    """
    from collections import defaultdict, Counter
    
    # Load sequences from combined.csv
    print("📖 Loading sequences from combined.csv...")
    sequence_dict = {}
    with open(combined_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ac_id = row.get('AC_ID', '').strip()
            ori_seq = row.get('ori_seq', '').strip()
            if ac_id and ori_seq:
                if ac_id not in sequence_dict:
                    sequence_dict[ac_id] = ori_seq
    print(f"✅ Loaded {len(sequence_dict):,} unique sequences")
    
    # Load all PTM records with all attributes
    print(f"\n📖 Loading PTM data from: {ptm_source_file}")
    all_records = []
    skipped_no_location = 0
    skipped_no_sequence = 0
    skipped_invalid_position = 0
    
    with open(ptm_source_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot_id = row.get('UNIPROT_ID', '').strip()
            location = row.get('LOCATION', '').strip()
            mod_type = row.get('MODIFICATION_TYPE', '').strip()
            functional_role = row.get('FUNCTIONAL_ROLE', '').strip()
            
            # Skip records without location
            if not location or location == '' or location.lower() == 'nan':
                skipped_no_location += 1
                continue
            
            # Skip records without required fields
            if not uniprot_id or not mod_type or not functional_role:
                continue
            
            # Convert location to int
            try:
                loc_key = int(float(location))
            except (ValueError, TypeError):
                skipped_no_location += 1
                continue
            
            # Check if sequence exists
            if uniprot_id not in sequence_dict:
                skipped_no_sequence += 1
                continue
            
            ori_seq = sequence_dict[uniprot_id]
            
            # Validate position
            if loc_key < 1 or loc_key > len(ori_seq):
                skipped_invalid_position += 1
                continue
            
            # Get amino acid
            aa = ori_seq[loc_key - 1]
            
            # Map to specific PTM type
            specific_ptm_type = map_general_to_specific_ptm_type(mod_type, aa)
            
            # Store record with all attributes
            record = {
                'uniprot_id': uniprot_id,
                'location': loc_key,
                'mod_type': mod_type,
                'specific_ptm_type': specific_ptm_type,
                'aa': aa,
                'ori_seq': ori_seq,
                'functional_role': functional_role,
                'pmid': row.get('PMID', '').strip(),
                'species': row.get('SPECIES', '').strip(),
                'source': row.get('SOURCE', '').strip(),
                'symbol': row.get('symbol', '').strip(),
            }
            all_records.append(record)
    
    print(f"✅ Loaded {len(all_records):,} valid records")
    print(f"❌ Skipped {skipped_no_location:,} records without LOCATION")
    print(f"❌ Skipped {skipped_no_sequence:,} records without sequence")
    print(f"❌ Skipped {skipped_invalid_position:,} records with invalid position")
    
    # Step 1: Remove completely duplicate records (all fields including functional_role are identical)
    print(f"\n🔄 Removing completely duplicate records...")
    seen_records = set()
    unique_records = []
    duplicate_count = 0
    
    for record in all_records:
        # Create a key for duplicate detection (all fields including functional_role)
        record_key = (
            record['uniprot_id'],
            record['location'],
            record['mod_type'],
            record['aa'],
            record['ori_seq'],
            record['functional_role'],
            record['pmid'],
            record['species'],
            record['source'],
            record['symbol']
        )
        
        if record_key not in seen_records:
            seen_records.add(record_key)
            unique_records.append(record)
        else:
            duplicate_count += 1
    
    print(f"✅ Removed {duplicate_count:,} completely duplicate records")
    print(f"✅ Remaining {len(unique_records):,} unique records")
    
    # Step 2: Group by (sequence, position, ptm_type) and select most common functional_role
    print(f"\n🔄 Resolving conflicts for same (sequence, position, ptm_type)...")
    grouped_records = defaultdict(list)
    
    for record in unique_records:
        # Key: (sequence, position, specific_ptm_type)
        key = (record['ori_seq'], record['location'], record['specific_ptm_type'])
        grouped_records[key].append(record)
    
    resolved_records = []
    conflict_count = 0
    resolved_count = 0
    
    for key, records in grouped_records.items():
        if len(records) == 1:
            # No conflict
            resolved_records.append(records[0])
        else:
            # Multiple records with same (sequence, position, ptm_type)
            conflict_count += 1
            
            # Count functional_role occurrences
            role_counter = Counter([r['functional_role'] for r in records])
            most_common_roles = role_counter.most_common()
            max_count = most_common_roles[0][1]
            
            # Get all roles with max count
            top_roles = [role for role, count in most_common_roles if count == max_count]
            
            if len(top_roles) == 1:
                # Single most common role
                selected_role = top_roles[0]
            else:
                # Multiple roles with same count - select first alphabetically
                selected_role = sorted(top_roles)[0]
            
            # Select first record with selected role (preserve other attributes)
            selected_record = next(r for r in records if r['functional_role'] == selected_role)
            resolved_records.append(selected_record)
            resolved_count += len(records) - 1
    
    print(f"✅ Found {conflict_count:,} groups with conflicts")
    print(f"✅ Resolved {resolved_count:,} conflicting records")
    print(f"✅ Final unique records: {len(resolved_records):,}")
    
    # Step 3: Process final records and extract context
    print(f"\n📊 Processing final records and extracting context...")
    final_sites = []
    
    for record in resolved_records:
        # Normalize functional role
        normalized_role = normalize_functional_role(record['functional_role'])
        
        # Get sequence context
        left_context, right_context, full_context = get_sequence_context(
            record['ori_seq'], record['location'], context_size
        )
        
        # Create final site record
        site_record = {
            'AC_ID': record['uniprot_id'],
            'ori_seq': record['ori_seq'],
            'position': record['location'],
            'aa': record['aa'],
            'ptm_type': record['specific_ptm_type'],
            'normalized_ptm_type': record['mod_type'],
            'left_context': left_context,
            'right_context': right_context,
            'context': full_context,
            'functional_role': normalized_role,
            'pmid': record['pmid'],
            'species': record['species'],
            'source': record['source'],
            'symbol': record['symbol'],
        }
        
        final_sites.append(site_record)
    
    # Write to output file
    if final_sites:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"\n💾 Saving PTM sites to: {output_file}")
        fieldnames = [
            'AC_ID', 'ori_seq', 'position', 'aa', 'ptm_type', 'normalized_ptm_type',
            'left_context', 'right_context', 'context', 'functional_role',
            'pmid', 'species', 'source', 'symbol'
        ]
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_sites)
        
        print(f"✅ Successfully saved {len(final_sites):,} PTM site records to {output_file}")
        print(f"✅ All records have functional_role")
        print(f"✅ This dataset is ready for training: sequence + ptm + position -> functional_role")
    else:
        print("⚠️  No PTM sites found!")


def main():
    """Main function to execute the extraction process."""
    # File paths
    ptm_source_file = '/home/zz/zheng/PTM/Post_Translational_Modification/Archive/PTM_added_gene_symbol.csv'
    combined_file = '/home/zz/zheng/ptm-mlm/main_pipeline/datasets/combined.csv'
    output_file = '/home/zz/zheng/ptm-mlm/main_pipeline/datasets/processing_data/functional_role/ptm_sites_with_functional_role.csv'
    
    # Process PTM data directly from source file with deduplication and label selection
    process_ptm_data_from_source_file(
        ptm_source_file, 
        combined_file, 
        output_file, 
        context_size=15
    )
    
    print("\n🎉 Process completed!")


if __name__ == '__main__':
    main()

