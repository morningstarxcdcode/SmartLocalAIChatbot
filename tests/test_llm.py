import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm import LocalLLM

def test_llama2_model_loading():
    # Replace 'meta-llama/Llama-2-7b-hf' with the actual LLaMA2 model name you want to test
    model_name = "meta-llama/Llama-2-7b-hf"
    llm = LocalLLM(model_name=model_name)
    assert llm.model is not None
    assert llm.tokenizer is not None

def test_generate_response():
    model_name = "meta-llama/Llama-2-7b-hf"
    llm = LocalLLM(model_name=model_name)
    prompt = "Hello, how are you?"
    response = llm.generate(prompt, max_length=50)
    assert isinstance(response, str)
    assert len(response) > 0
