import json
from pathlib import Path


class TrainLogger:
    """
    Logger that saves training metrics and config to JSON files.
    """
    def __init__(self, save_dir=None, config=None, resume=False):
        """
        Initialize logger.
        @param save_dir: Directory to save metrics and config.
        @param config: Configuration dict to save.
        @param resume: If True, load existing metrics from metrics.json if it exists (for resuming training).
        """
        self.save_dir = Path(save_dir) if save_dir else None
        self.metrics = []
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path = self.save_dir / "metrics.json"
            
            # Load existing metrics if resuming training
            if resume and self.metrics_path.exists():
                try:
                    with open(self.metrics_path, 'r', encoding='utf-8') as f:
                        self.metrics = json.load(f)
                    print(f"📊 Loaded {len(self.metrics)} existing metrics from {self.metrics_path}")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"⚠️  Warning: Could not load existing metrics: {e}. Starting with empty metrics.")
                    self.metrics = []

    def log(self, data):
        """
        Log training metrics (accumulated in memory, not written to disk).
        @param data: Dictionary of metrics to log.
        """
        # Round float values for readability
        data = {k: round(v, 4) if isinstance(v, float) else v for k, v in data.items()}
        
        # Print to console
        print(data)
        
        # Accumulate in memory only
        self.metrics.append(data)
    
    def finalize(self):
        """
        Write all accumulated metrics to file (call this at the end of training).
        Includes both existing metrics (if resuming) and newly logged metrics.
        """
        if self.save_dir and len(self.metrics) > 0:
            with open(self.metrics_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            print(f"💾 Metrics saved to: {self.metrics_path} (Total: {len(self.metrics)} entries)")
