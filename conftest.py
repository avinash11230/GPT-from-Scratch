"""
pytest conftest.py — adds the project root to sys.path so test modules can
import project code (model, tokenizer, config, etc.) without any installation.
"""

import sys
import os

# Ensure the project root is on the path regardless of how pytest is invoked
sys.path.insert(0, os.path.dirname(__file__))
