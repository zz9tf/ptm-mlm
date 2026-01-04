"""
Script to generate CSV summary files from existing length group directories.
This script scans all length_* directories and creates CSV files for validation and test metrics.
"""
import os
import json
import csv
from pathlib import Path
from collections import defaultdict


def write_metrics_to_csv(summary: dict, output_path: Path, metrics_type: str = 'test'):
    """
    Write metrics to CSV file.
    
    @param summary: Summary dictionary containing length groups data
    @param output_path: Output directory path
    @param metrics_type: Type of metrics to export ('test' or 'validation')
    """
    csv_filename = f"length_groups_summary_{metrics_type}.csv"
    csv_path = output_path / csv_filename
    
    # Define CSV columns
    columns = ['Length', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC', 'MCC']
    
    # Write CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(columns)
        
        # Write data rows
        for seq_len in sorted(summary['length_groups'].keys(), key=int):
            group = summary['length_groups'][seq_len]
            
            if metrics_type == 'test':
                metrics = group.get('test_metrics', {})
            else:
                metrics = group.get('best_validation_metrics', {})
            
            if metrics:
                # Format numbers with 4 decimal places for CSV
                row = [
                    int(seq_len),
                    round(metrics.get('loss'), 4) if metrics.get('loss') is not None else "N/A",
                    round(metrics.get('accuracy'), 4) if metrics.get('accuracy') is not None else "N/A",
                    round(metrics.get('precision'), 4) if metrics.get('precision') is not None else "N/A",
                    round(metrics.get('recall'), 4) if metrics.get('recall') is not None else "N/A",
                    round(metrics.get('f1'), 4) if metrics.get('f1') is not None else "N/A",
                    round(metrics.get('auroc'), 4) if metrics.get('auroc') is not None else "N/A",
                    round(metrics.get('auprc'), 4) if metrics.get('auprc') is not None else "N/A",
                    round(metrics.get('mcc'), 4) if metrics.get('mcc') is not None else "N/A"
                ]
            else:
                row = [int(seq_len)] + ['N/A'] * 8
            
            writer.writerow(row)
    
    print(f"💾 {metrics_type.capitalize()} metrics CSV saved to: {csv_path}")
    return csv_path


def generate_summary(output_dir: str):
    """
    Generate CSV summary files from existing length group directories.
    Creates CSV files for validation and test metrics.
    
    @param output_dir: Output directory containing length_* subdirectories
    @return: Path to the test metrics CSV file
    """
    output_path = Path(output_dir)
    
    # Find all length_* directories
    length_dirs = sorted([d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('length_')])
    
    if not length_dirs:
        print(f"⚠️  No length_* directories found in {output_dir}")
        return
    
    print(f"📊 Found {len(length_dirs)} length group directories")
    
    summary = {
        'training_mode': 'by_length_groups',
        'total_length_groups': len(length_dirs),
        'length_groups': {}
    }
    
    for length_dir in length_dirs:
        # Extract sequence length from directory name (e.g., "length_61" -> 61)
        try:
            seq_len = int(length_dir.name.split('_')[1])
        except (IndexError, ValueError):
            print(f"⚠️  Cannot parse length from directory name: {length_dir.name}")
            continue
        
        # Try to load evaluation_results.json
        eval_results_path = length_dir / "evaluation_results.json"
        
        if eval_results_path.exists():
            try:
                with open(eval_results_path, 'r') as f:
                    results = json.load(f)
                
                # Extract best validation metrics
                best_val_metrics = results.get('best_validation_metrics', {})
                
                # Extract test metrics
                test_metrics = results.get('test_metrics', {})
                
                summary['length_groups'][seq_len] = {
                    'output_dir': str(length_dir),
                    'best_validation_metrics': {
                        'loss': best_val_metrics.get('loss'),
                        'accuracy': best_val_metrics.get('accuracy'),
                        'precision': best_val_metrics.get('precision'),
                        'recall': best_val_metrics.get('recall'),
                        'f1': best_val_metrics.get('f1'),
                        'auroc': best_val_metrics.get('auroc'),
                        'auprc': best_val_metrics.get('auprc'),
                        'mcc': best_val_metrics.get('mcc'),
                        'epoch': best_val_metrics.get('epoch')
                    },
                    'test_metrics': {
                        'loss': test_metrics.get('loss'),
                        'accuracy': test_metrics.get('accuracy'),
                        'precision': test_metrics.get('precision'),
                        'recall': test_metrics.get('recall'),
                        'f1': test_metrics.get('f1'),
                        'auroc': test_metrics.get('auroc'),
                        'auprc': test_metrics.get('auprc'),
                        'mcc': test_metrics.get('mcc'),
                        'confusion_matrix': test_metrics.get('confusion_matrix', {})
                    }
                }
                print(f"✅ Loaded results for length {seq_len}")
            except Exception as e:
                print(f"⚠️  Error loading {eval_results_path}: {e}")
                summary['length_groups'][seq_len] = {
                    'output_dir': str(length_dir),
                    'best_validation_metrics': None,
                    'test_metrics': None,
                    'error': str(e)
                }
        else:
            print(f"⚠️  evaluation_results.json not found in {length_dir}")
            summary['length_groups'][seq_len] = {
                'output_dir': str(length_dir),
                'best_validation_metrics': None,
                'test_metrics': None,
                'note': 'evaluation_results.json not found'
            }
    
    # Find best model based on AUPRC and MCC (user's preferred metrics)
    # Priority: AUPRC first, then MCC as tiebreaker
    best_length = None
    best_auprc = None
    best_mcc = None
    best_model_info = None
    
    for seq_len, group in summary['length_groups'].items():
        test_metrics = group.get('test_metrics', {})
        if test_metrics:
            auprc = test_metrics.get('auprc')
            mcc = test_metrics.get('mcc')
            
            if auprc is not None:
                # Select based on AUPRC first, then MCC as tiebreaker
                if best_auprc is None:
                    best_auprc = auprc
                    best_mcc = mcc if mcc is not None else 0
                    best_length = int(seq_len)
                    best_model_info = {
                        'length': int(seq_len),
                        'output_dir': group.get('output_dir'),
                        'best_validation_metrics': group.get('best_validation_metrics'),
                        'test_metrics': group.get('test_metrics')
                    }
                elif auprc > best_auprc:
                    # Better AUPRC
                    best_auprc = auprc
                    best_mcc = mcc if mcc is not None else 0
                    best_length = int(seq_len)
                    best_model_info = {
                        'length': int(seq_len),
                        'output_dir': group.get('output_dir'),
                        'best_validation_metrics': group.get('best_validation_metrics'),
                        'test_metrics': group.get('test_metrics')
                    }
                elif auprc == best_auprc and mcc is not None:
                    # Same AUPRC, use MCC as tiebreaker
                    if best_mcc is None or mcc > best_mcc:
                        best_mcc = mcc
                        best_length = int(seq_len)
                        best_model_info = {
                            'length': int(seq_len),
                            'output_dir': group.get('output_dir'),
                            'best_validation_metrics': group.get('best_validation_metrics'),
                            'test_metrics': group.get('test_metrics')
                        }
    
    # Add best model to summary (based on AUPRC and MCC)
    if best_model_info:
        summary['best_model'] = {
            'selection_criterion': 'test_auprc_and_mcc',
            'test_auprc': best_auprc,
            'test_mcc': best_mcc,
            **best_model_info
        }
    
    # Export to CSV files
    print(f"\n💾 Total length groups: {len(summary['length_groups'])}")
    validation_csv_path = write_metrics_to_csv(summary, output_path, 'validation')
    test_csv_path = write_metrics_to_csv(summary, output_path, 'test')
    
    # Print best model summary
    if best_model_info:
        print("\n🏆 Best Model (Selected by Test AUPRC and MCC):")
        print("-" * 70)
        print(f"   Length: {best_length}")
        print(f"   Test AUPRC: {best_auprc:.4f}")
        print(f"   Test MCC: {best_mcc:.4f}")
        test_metrics = best_model_info.get('test_metrics', {})
        if test_metrics:
            print(f"   Test Metrics:")
            print(f"      Loss: {test_metrics.get('loss', 'N/A'):.4f}" if test_metrics.get('loss') else "      Loss: N/A")
            print(f"      Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}" if test_metrics.get('accuracy') else "      Accuracy: N/A")
            print(f"      Precision: {test_metrics.get('precision', 'N/A'):.4f}" if test_metrics.get('precision') else "      Precision: N/A")
            print(f"      Recall: {test_metrics.get('recall', 'N/A'):.4f}" if test_metrics.get('recall') else "      Recall: N/A")
            print(f"      F1: {test_metrics.get('f1', 'N/A'):.4f}" if test_metrics.get('f1') else "      F1: N/A")
            print(f"      AUROC: {test_metrics.get('auroc', 'N/A'):.4f}" if test_metrics.get('auroc') else "      AUROC: N/A")
            print(f"      AUPRC: {test_metrics.get('auprc', 'N/A'):.4f}" if test_metrics.get('auprc') else "      AUPRC: N/A")
            print(f"      MCC: {test_metrics.get('mcc', 'N/A'):.4f}" if test_metrics.get('mcc') else "      MCC: N/A")
    
    # Print summary statistics
    print("\n📊 Summary Statistics - Best Validation Metrics:")
    print("-" * 100)
    header = f"{'Length':<8} {'Loss':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'AUROC':<10} {'AUPRC':<10} {'MCC':<10}"
    print(header)
    print("-" * 100)
    
    for seq_len in sorted(summary['length_groups'].keys()):
        group = summary['length_groups'][seq_len]
        val_metrics = group.get('best_validation_metrics', {})
        
        if val_metrics:
            loss = val_metrics.get('loss')
            acc = val_metrics.get('accuracy')
            prec = val_metrics.get('precision')
            rec = val_metrics.get('recall')
            f1 = val_metrics.get('f1')
            auroc = val_metrics.get('auroc')
            auprc = val_metrics.get('auprc')
            mcc = val_metrics.get('mcc')
            
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            prec_str = f"{prec:.4f}" if prec is not None else "N/A"
            rec_str = f"{rec:.4f}" if rec is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"
            mcc_str = f"{mcc:.4f}" if mcc is not None else "N/A"
            
            print(f"{seq_len:<8} {loss_str:<10} {acc_str:<12} {prec_str:<12} {rec_str:<12} {f1_str:<10} {auroc_str:<10} {auprc_str:<10} {mcc_str:<10}")
        else:
            print(f"{seq_len:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print("\n📊 Summary Statistics - Test Metrics:")
    print("-" * 100)
    print(header)
    print("-" * 100)
    
    for seq_len in sorted(summary['length_groups'].keys()):
        group = summary['length_groups'][seq_len]
        test_metrics = group.get('test_metrics', {})
        
        if test_metrics:
            loss = test_metrics.get('loss')
            acc = test_metrics.get('accuracy')
            prec = test_metrics.get('precision')
            rec = test_metrics.get('recall')
            f1 = test_metrics.get('f1')
            auroc = test_metrics.get('auroc')
            auprc = test_metrics.get('auprc')
            mcc = test_metrics.get('mcc')
            
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            prec_str = f"{prec:.4f}" if prec is not None else "N/A"
            rec_str = f"{rec:.4f}" if rec is not None else "N/A"
            f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
            auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
            auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"
            mcc_str = f"{mcc:.4f}" if mcc is not None else "N/A"
            
            print(f"{seq_len:<8} {loss_str:<10} {acc_str:<12} {prec_str:<12} {rec_str:<12} {f1_str:<10} {auroc_str:<10} {auprc_str:<10} {mcc_str:<10}")
        else:
            print(f"{seq_len:<8} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    return test_csv_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CSV summary files from existing length group directories")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory containing length_* subdirectories"
    )
    
    args = parser.parse_args()
    
    generate_summary(args.output_dir)

