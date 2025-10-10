# src/biocircuitai/io/logging.py
from __future__ import annotations
import logging, sys, os
from datetime import datetime

def get_logger(name: str = "biocircuitai", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # optional file handler (rotated by date prefix)
    os.makedirs("logs", exist_ok=True)
    fname = f"logs/{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(fname, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
