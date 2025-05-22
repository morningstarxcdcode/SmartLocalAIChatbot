import os
import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

try:
    import faiss
except ImportError:
    faiss = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class VectorStore:
    def __init__(self, config: dict):
        self.embedding_model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store_type = config.get("vector_store", "faiss")
        self.persist_directory = config.get("persist_directory", "./data/vector_store")
        os.makedirs(self.persist_directory, exist_ok=True)

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.documents = []
        self.ids = []

        if self.vector_store_type == "faiss":
            if faiss is None:
                raise ImportError("faiss is not installed. Please install faiss-cpu.")
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.index_path = os.path.join(self.persist_directory, "faiss.index")
            self._load_faiss_index()
        elif self.vector_store_type == "chromadb":
            if chromadb is None:
                raise ImportError("chromadb is not installed. Please install chromadb.")
            self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory))
            self.collection = self.client.get_or_create_collection(name="documents")
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

    def _load_faiss_index(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def _save_faiss_index(self):
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def add_documents(self, documents: List[str]):
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        if self.vector_store_type == "faiss":
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            self._save_faiss_index()
            self.documents.extend(documents)
        elif self.vector_store_type == "chromadb":
            ids = [str(len(self.documents) + i) for i in range(len(documents))]
            self.collection.add(documents=documents, embeddings=embeddings.tolist(), ids=ids)
            self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to vector store")

    def query(self, query_text: str, top_k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)
        if self.vector_store_type == "faiss":
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            distances, indices = self.index.search(query_embedding, top_k)
            results = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            return results
        elif self.vector_store_type == "chromadb":
            results = self.collection.query(query_texts=[query_text], n_results=top_k)
            return [doc for doc in results['documents'][0]]
        else:
            return []
