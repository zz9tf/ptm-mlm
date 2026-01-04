#!/bin/bash
#
# Script to query drug target labels from ChEMBL for dbPTM sequences.
# This script runs the query_drug_labels.py script with appropriate parameters.
#

source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate ptm-mamba

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Default parameters
INPUT_FILE="${1:-dbptm_data.csv}"
OUTPUT_FILE="${2:-}"
ACCESSION_COL="${3:-AC_ID}"
SLEEP_TIME="${4:-0.05}"

echo "=========================================="
echo "🚀 Starting Drug Target Label Query"
echo "=========================================="
echo "Input file: $INPUT_FILE"
echo "Output file: ${OUTPUT_FILE:-auto-generated}"
echo "Accession column: $ACCESSION_COL"
echo "Sleep time: $SLEEP_TIME seconds"
echo "=========================================="
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Run the query script
echo "📡 Starting query process..."
echo "   (This may take a very long time)"
echo ""

python query_drug_labels.py "$INPUT_FILE" "$OUTPUT_FILE" "$ACCESSION_COL" "$SLEEP_TIME"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Query completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Query failed with exit code: $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi

