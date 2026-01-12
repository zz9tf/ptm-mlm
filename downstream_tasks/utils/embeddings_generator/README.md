# Embeddings Generator

This directory contains scripts to generate embeddings for all downstream tasks from pre-trained models.

## Directory Structure

The generated embeddings follow this structure:
```
embeddings/
├── {model_name}_{layer}/
│   ├── p_site/
│   │   ├── train_embeddings.pt
│   │   ├── train_labels.pt
│   │   ├── train_sequences.pt
│   │   ├── valid_embeddings.pt
│   │   ├── valid_labels.pt
│   │   ├── valid_sequences.pt
│   │   ├── test_embeddings.pt
│   │   ├── test_labels.pt
│   │   └── test_sequences.pt
│   ├── nhas/
│   │   └── ... (same structure)
│   └── ppi/
│       ├── train_binder_embeddings.pt
│       ├── train_wt_embeddings.pt
│       ├── train_ptm_embeddings.pt
│       ├── train_labels.pt
│       ├── valid_binder_embeddings.pt
│       ├── valid_wt_embeddings.pt
│       ├── valid_ptm_embeddings.pt
│       ├── valid_labels.pt
│       ├── test_binder_embeddings.pt
│       ├── test_wt_embeddings.pt
│       ├── test_ptm_embeddings.pt
│       └── test_labels.pt
```

## Model Name Format

Model names are converted to directory names using these rules:
- `facebook/esm2_t30_150M_UR50D` → `esm2-t30_150M_UR50D`
- `facebook/esm2_t33_650M_UR50D` → `esm2-t33_650M_UR50D`

Layer suffixes:
- Last layer: `_last`
- Specific layer: `_layer{N}`

## Usage

### Generate embeddings for a specific model and layer

```bash
# Generate embeddings for ESM2-150M last layer
python p_site_generator.py --pretrained_model_name facebook/esm2_t30_150M_UR50D --output_dir /path/to/embeddings

# Generate embeddings for ESM2-150M layer 5
python p_site_generator.py --pretrained_model_name facebook/esm2_t30_150M_UR50D --layer_index 5 --output_dir /path/to/embeddings
```

## Available Generators

1. **p_site_generator.py**: Phosphorylation site prediction
2. **nhas_generator.py**: NHA site prediction
3. **ppi_generator.py**: Protein-protein interaction prediction

## Parameters

All generators support these parameters:

- `--pretrained_model_name`: HuggingFace model name (default: facebook/esm2_t30_150M_UR50D)
- `--layer_index`: Layer index to extract (0-based, None for last layer)
- `--output_dir`: Base output directory (default: /home/zz/zheng/ptm-mlm/downstream_tasks/embeddings)
- `--batch_size`: Batch size for inference (default: 32)
- `--max_sequence_length`: Maximum sequence length
- `--use_sliding_window`: Use sliding window for long sequences

Task-specific parameters are also available in each generator.

## Integration with Downstream Tasks

Downstream tasks can load embeddings using this pattern:

```python
import torch
import os

def load_embeddings(model_name, layer_index, task_name, split_name, output_dir):
    model_short_name = model_name.replace('/', '_').replace('facebook_', '').replace('esm2_', 'esm2-').replace('esm_', 'esm-')
    layer_suffix = f"layer{layer_index}" if layer_index is not None else "last"
    model_layer_dir = f"{model_short_name}_{layer_suffix}"

    task_dir = os.path.join(output_dir, model_layer_dir, task_name)

    embeddings = torch.load(os.path.join(task_dir, f"{split_name}_embeddings.pt"))
    labels = torch.load(os.path.join(task_dir, f"{split_name}_labels.pt"))
    sequences = torch.load(os.path.join(task_dir, f"{split_name}_sequences.pt"))

    return embeddings, labels, sequences
```

## Adding New Tasks

To add a new downstream task:

1. Create `{task_name}_generator.py` in this directory
2. Follow the same parameter structure and directory naming conventions
3. Update the bash scripts to include the new generator
4. Update this README

The embeddings generator will be expanded as new downstream tasks are added.
