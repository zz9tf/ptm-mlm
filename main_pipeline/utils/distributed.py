import os


def is_main_process_from_env():
    """
    Check if current process is main process (rank 0) by checking environment variables.
    This is called BEFORE Accelerator initialization, so we check env vars set by accelerate/torchrun.
    
    @returns: True if this is the main process (rank 0), False otherwise.
    """
    # Check common environment variables set by distributed training frameworks
    local_rank = os.environ.get("LOCAL_RANK", None)
    rank = os.environ.get("RANK", None)
    
    # If neither is set, assume single process (main process)
    if local_rank is None and rank is None:
        return True
    
    # If LOCAL_RANK or RANK is 0, this is the main process
    if local_rank is not None:
        return int(local_rank) == 0
    if rank is not None:
        return int(rank) == 0
    
    return True  # Default to True if unclear

