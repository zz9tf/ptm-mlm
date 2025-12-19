import yaml

def load_config(config_path):
    """
    Load configuration from YAML file and convert numeric strings to proper types.
    @param config_path: Path to YAML configuration file.
    @returns: Configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Recursively convert numeric strings to floats/ints
    def convert_numeric(obj):
        if isinstance(obj, dict):
            return {k: convert_numeric(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numeric(item) for item in obj]
        elif isinstance(obj, str):
            # Try to convert scientific notation strings to float
            try:
                # Handle scientific notation like "4e-4"
                if 'e' in obj.lower() or 'E' in obj:
                    return float(obj)
                # Try int first, then float
                if '.' in obj:
                    return float(obj)
                return int(obj)
            except (ValueError, TypeError):
                return obj
        return obj
    
    return convert_numeric(config)

