"""Put ``src/`` on the path so the examples run from a checkout without install."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

CORPUS = os.path.join(REPO_ROOT, "data", "corpus")
ADVERSARIAL = os.path.join(REPO_ROOT, "data", "adversarial")
GOLDSET = os.path.join(REPO_ROOT, "data", "goldset.json")
