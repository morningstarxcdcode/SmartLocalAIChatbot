import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.vector_store import VectorStore

def test_dummy_vector_store():
    config = {}  # Provide minimal config if needed
    vs = VectorStore(config)
    assert vs is not None
