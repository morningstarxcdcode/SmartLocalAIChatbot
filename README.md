# SmartLocalAIChatbot

## Project Overview

Smart Local AI Chatbot with Model Optimization Toolkit

This is a modular, production-ready AI/ML chatbot with support for:

- Local LLMs (like LLaMA2, GPT-J, etc.)
- PDF/text knowledge base
- Vector search (FAISS/ChromaDB)
- Flask Web UI + CLI
- Few-shot learning
- ML Model Optimization Toolkit (for MLOps, quantization, pruning, ONNX, etc.)

## Folder Structure

```
SmartLocalAIChatbot/
├── main.py                         # CLI entry point
├── requirements.txt                # Dependencies
├── README.md                      # Project documentation

├── config/
│   └── config.yaml                # Config file with paths, model settings, etc.

├── interface/
│   └── app.py                    # Flask web chat UI

├── models/
│   └── llm.py                    # LLM loading (LLaMA2, GPT-J, etc.)

├── utils/
│   ├── pdf_loader.py             # Load PDFs and text files
│   ├── vector_store.py           # FAISS or ChromaDB integration
│   └── few_shot.py               # Prompt injection or few-shot learning

├── ml_opt/                       # Model Optimization Toolkit
│   ├── train_model.py            # Train tiny model (e.g. DistilBERT or CNN)
│   ├── optimize_model.py         # Quantize, prune, compress
│   ├── convert_to_onnx.py        # ONNX conversion script
│   ├── benchmark.py              # CPU vs GPU benchmarking
│   ├── pipeline_runner.py        # Automate the optimization pipeline
│   ├── model_utils.py            # Helpers for model saving/loading
│   └── notebooks/
│       └── demo_pipeline.ipynb   # Jupyter demo of optimization pipeline

├── data/                        # User-provided documents or training data
│   └── sample.pdf               # Sample for loading and vectorizing

├── tests/
│   ├── test_llm.py              # Test for LLM functions
│   ├── test_vector_store.py     # Test vector database integration
│   ├── test_few_shot.py         # Test prompt tuning
│   └── test_pipeline.py         # Test model optimization workflow
```

## How to Start the Project

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd SmartLocalAIChatbot
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variable for Flask app:**

   ```bash
   export FLASK_APP=interface/app.py
   ```

4. **Run the Flask web UI:**

   ```bash
   flask run
   ```

   The web UI will be available at `http://127.0.0.1:5000/`.

5. **Using the CLI:**

   Run the main CLI entry point:

   ```bash
   python main.py
   ```

## Important Notes

- Ensure you have Python 3.11 or higher installed.
- The project supports GPU acceleration if available.
- The LocalLLM model is initialized with a default dummy model; update `interface/app.py` to use your desired model and path.
- The ML Optimization Toolkit (`ml_opt/`) contains scripts for training, optimizing, and benchmarking models.
- Tests are available in the `tests/` folder; run them with:

  ```bash
  pytest
  ```

- Dockerfile and GitHub Actions workflow are included for advanced deployment and CI/CD.

## Contributing

We welcome contributions to improve SmartLocalAIChatbot! To contribute:

- Fork the repository and create your feature branch.
- Write clear, concise commit messages.
- Ensure all tests pass before submitting a pull request.
- Follow the existing code style and conventions.
- Open an issue to discuss major changes before implementation.

## Contact

For issues or contributions, please open an issue or pull request on the GitHub repository.
