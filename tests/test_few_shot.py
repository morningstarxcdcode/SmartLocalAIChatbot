import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.few_shot import FewShotTrainer

def test_dummy_few_shot():
    config = {}  # Provide minimal config if needed
    fst = FewShotTrainer(config)
    assert fst is not None
