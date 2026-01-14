"""
NeuroAlign
==========

Align brains through regional age patterns - Find your brain twins.

This package implements a pipeline for:
1. Loading multi-modal neuroimaging data
2. Training regional brain age prediction models  
3. Computing regional Brain Age Gap (BAG) vectors
4. Building embedding spaces for similarity retrieval
5. LLM-powered interpretation of brain similarity patterns
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import data, embedding, modeling, retrieval, utils, visualization

__all__ = [
    "data",
    "embedding",
    "modeling",
    "retrieval",
    "utils",
    "visualization",
]
