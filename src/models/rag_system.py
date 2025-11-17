"""Lightweight RAG system using ChromaDB and Claude."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

from src.config import RAGConfig
from src.models.base_model import BaseMLModel


class RAGSystem(BaseMLModel):
    """
    Retrieval-Augmented Generation system.

    Components:
    - Embedding: all-MiniLM-L6-v2 (~80MB)
    - Vector DB: ChromaDB (local)
    - LLM: Claude via Anthropic API
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG system.

        Args:
            config: RAGConfig object or None for defaults
        """
        if config is None:
            config = RAGConfig()
        super().__init__(config)

        # Initialize embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(config.embedding_model)

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        try:
            self.collection = self.chroma_client.get_collection(config.collection_name)
            print(f"Loaded existing collection: {config.collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(config.collection_name)
            print(f"Created new collection: {config.collection_name}")

        # Initialize Claude
        self.claude = Anthropic(api_key=config.anthropic_api_key)

        self._is_trained = False  # RAG is "trained" when documents are ingested

    def train(
            self, documents: List[str], metadata: Optional[List[dict]] = None
    ) -> Dict[str, float]:
        """
        'Train' the RAG system by ingesting documents.

        Args:
            documents: List of text documents to ingest
            metadata: Optional metadata for each document

        Returns:
            Dict with ingestion metrics
        """
        return self.ingest_documents(documents, metadata)

    def ingest_documents(
        self, documents: List[str], metadata: Optional[List[dict]] = None
    ) -> Dict[str, float]:
        """
        Ingest documents into the vector database.

        Args:
            documents: List of text documents
            metadata: Optional metadata for each document

        Returns:
            Dict with ingestion metrics
        """
        if not documents:
            raise ValueError("No documents provided for ingestion")

        start_time = time.time()

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedder.encode(documents, show_progress_bar=True)

        # Prepare metadata
        if metadata is None:
            metadata = [{"index": i} for i in range(len(documents))]

        # Add to ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata,  # type: ignore
            ids=ids
        )

        ingestion_time = time.time() - start_time
        self._is_trained = True

        metrics = {
            "num_documents": len(documents),
            "ingestion_time_sec": round(ingestion_time, 2),
            "avg_doc_length": sum(len(doc) for doc in documents) / len(documents),
        }

        self.metrics.update(metrics)
        return metrics

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if top_k is None:
            top_k = self.config.top_k

        # Generate query embedding
        query_embedding = self.embedder.encode([query])

        # Search in ChromaDB
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)

        return results["documents"][0] if results["documents"] else []

    def generate(self, query: str, context: List[str]) -> str:
        """
        Generate answer using Claude with retrieved context.

        Args:
            query: User query
            context: Retrieved context documents

        Returns:
            Generated answer
        """
        # Construct prompt
        context_text = "\n\n".join(
            f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)
        )

        prompt = f"""
        Based on the following context documents, please answer the question.
        If the answer cannot be found in the context, say so.

        Context:
        {context_text}

        Question: {query}

        Answer:
        """

        # Call Claude API
        message = self.claude.messages.create(  # type: ignore
            model=self.config.claude_model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text  # type: ignore

    def predict(self, query: str) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve + generate.

        Args:
            query: User query

        Returns:
            Dict with answer, context, and metadata
        """
        start_time = time.time()

        # Retrieve relevant documents
        context = self.retrieve(query)

        # Generate answer
        answer = self.generate(query, context)

        latency = (time.time() - start_time) * 1000  # Convert to ms

        return {"answer": answer, "context": context, "latency_ms": round(latency, 2)}

    def evaluate(self, test_queries: List[Tuple[str, List[str]]]) -> Dict[str, float]:
        """
        Evaluate RAG system on test queries.

        Args:
            test_queries: List of (query, expected_docs) tuples

        Returns:
            Dict with evaluation metrics
        """
        if not test_queries:
            raise ValueError("No test queries provided")

        precision_scores = []
        latencies = []

        for query, expected_docs in test_queries:
            start_time = time.time()
            retrieved_docs = self.retrieve(query)
            latency = (time.time() - start_time) * 1000

            # Calculate precision
            retrieved_set = set(retrieved_docs)
            expected_set = set(expected_docs)
            precision = (
                len(retrieved_set & expected_set) / len(retrieved_set)
                if retrieved_set
                else 0
            )

            precision_scores.append(precision)
            latencies.append(latency)

        metrics = {
            "retrieval_precision": sum(precision_scores) / len(precision_scores),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "num_test_queries": len(test_queries),
        }

        self.metrics.update(metrics)
        return metrics

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        count = self.collection.count()
        return {"total_documents": count, "collection_name": self.config.collection_name}

    def _save_implementation(self, path: Path) -> None:
        """
        Save RAG system state.

        Note: ChromaDB persists automatically. We save minimal state here.
        """
        import json

        state = {
            "config": self.config.__dict__,
            "collection_name": self.config.collection_name,
            "num_documents": self.collection.count(),
        }

        # Filter out non-serializable objects
        state["config"] = {
            k: v for k, v in state["config"].items() if isinstance(v, (str, int, float, bool))
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_implementation(self, path: Path) -> None:
        """
        Load RAG system state.

        Note: ChromaDB collection should already exist.
        """
        import json

        with open(path, "r") as f:
            state = json.load(f)

        # Verify collection exists
        try:
            self.collection = self.chroma_client.get_collection(state["collection_name"])
            self._is_trained = True
        except Exception as e:
            raise ValueError(f"Collection '{state['collection_name']}' not found: {e}")


if __name__ == "__main__":
    # Quick test
    config = RAGConfig()
    rag = RAGSystem(config)

    # Ingest sample documents
    docs = [
        "The capital of France is Paris.",
        "Python is a popular programming language.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    metrics = rag.ingest_documents(docs)
    print(f"Ingestion metrics: {metrics}")

    # Test query
    result = rag.predict("What is the capital of France?")
    print(f"\nQuery result: {result}")
