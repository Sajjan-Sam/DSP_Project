import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def get_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
    logger.log_section = lambda msg: logger.info(f"\n{'-'*70}\n{msg}\n{'-'*70}")
    logger.start_timer = lambda name: setattr(logger, f'_timer_{name}', datetime.now())
    logger.end_timer = lambda name: logger.info(
        f"Timer '{name}': {(datetime.now() - getattr(logger, f'_timer_{name}', datetime.now())).total_seconds():.2f}s"
    ) if hasattr(logger, f'_timer_{name}') else None
    logger.finalize = lambda: logger.info("Execution completed")
    
    return logger