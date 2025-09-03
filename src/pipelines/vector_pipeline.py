"""
Vector Pipeline for medical RAG chatbot.

Handles:
- Embedding model setup
- Pinecone vector database operations
- Document upsertion
"""

import os
import time
import mlflow
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from typing import List
from langchain.schema import Document


class VectorPipeline:
    """Pipeline for vector database operations."""

    def __init__(self, pinecone_api_key: str = None):
        """
        Initialize the vector pipeline.

        Args:
            pinecone_api_key: Pinecone API key (will try to get from env if not provided)
        """
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key is required")

        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.embedding = None
        self.index = None

    def setup_embeddings(self, model_name: str = "nomic-embed-text:latest",
                         base_url: str = "http://localhost:11434") -> OllamaEmbeddings:
        """
        Set up the embedding model.

        Args:
            model_name: Name of the Ollama embedding model
            base_url: Base URL for Ollama instance

        Returns:
            Configured OllamaEmbeddings object
        """
        # Log embedding model parameters
        mlflow.log_param("embedding_model_name", model_name)
        mlflow.log_param("embedding_provider", "Ollama")
        mlflow.log_param("embedding_class", "OllamaEmbeddings")
        mlflow.log_param("ollama_base_url", base_url)

        start_time = time.time()
        embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        end_time = time.time()
        loading_duration = end_time - start_time

        # Log metrics
        mlflow.log_metric("embedding_loading_duration_seconds", loading_duration)

        print(f"Connected to Ollama embedding model '{model_name}' at {base_url} (init {loading_duration:.2f}s).")

        self.embedding = embeddings
        return embeddings

    def setup_pinecone_index(self, index_name: str, dimension: int,
                            cloud_provider: str = "aws", region: str = "us-east-1") -> None:
        """
        Set up Pinecone index.

        Args:
            index_name: Name of the Pinecone index
            dimension: Dimension of the embeddings
            cloud_provider: Cloud provider for serverless spec
            region: Region for serverless spec
        """
        # Log stage name
        mlflow.log_param("stage", "pinecone_index_setup")
        mlflow.log_param("pinecone_index_name", index_name)
        mlflow.log_param("pinecone_index_dimension", dimension)
        mlflow.log_param("pinecone_index_metric", "cosine")
        mlflow.log_param("pinecone_cloud_provider", cloud_provider)
        mlflow.log_param("pinecone_region", region)

        # Check if index exists
        start_check_time = time.time()
        index_exists = self.pc.has_index(index_name)
        end_check_time = time.time()
        check_duration = end_check_time - start_check_time

        mlflow.log_metric("pinecone_index_exists_check_duration_seconds", check_duration)
        mlflow.log_param("pinecone_index_existed_before_run", index_exists)

        if not index_exists:
            mlflow.log_param("pinecone_index_action", "created_new_index")
            print(f"Pinecone index '{index_name}' does not exist. Creating...")
            start_create_time = time.time()
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud=cloud_provider, region=region)
            )
            end_create_time = time.time()
            creation_duration = end_create_time - start_create_time
            mlflow.log_metric("pinecone_index_creation_duration_seconds", creation_duration)
            print(f"Pinecone index '{index_name}' created in {creation_duration:.2f} seconds.")
        else:
            mlflow.log_param("pinecone_index_action", "connected_to_existing_index")
            print(f"Pinecone index '{index_name}' already exists. Connecting...")
            mlflow.log_metric("pinecone_index_creation_duration_seconds", 0)

        # Connect to the index
        start_connect_time = time.time()
        self.index = self.pc.Index(index_name)
        end_connect_time = time.time()
        connect_duration = end_connect_time - start_connect_time

        mlflow.log_metric("pinecone_index_connection_duration_seconds", connect_duration)
        print(f"Connected to Pinecone index '{index_name}' in {connect_duration:.4f} seconds.")

        # Log index stats
        try:
            index_info = self.index.describe_index_stats()
            mlflow.log_metric("pinecone_total_vector_count", index_info.dimension)
            if index_info.namespaces:
                total_vectors_in_index = sum(ns.vector_count for ns_name, ns in index_info.namespaces.items())
                mlflow.log_metric("pinecone_total_vectors_in_index", total_vectors_in_index)
                mlflow.log_param("pinecone_namespaces", list(index_info.namespaces.keys()))
        except Exception as e:
            print(f"Could not get Pinecone index stats: {e}")
            mlflow.log_param("pinecone_index_stats_error", str(e))

    def upsert_documents(self, chunks: List[Document], index_name: str,
                        batch_size: int = 50) -> None:
        """
        Upsert documents to Pinecone index.

        Args:
            chunks: List of Document objects to upsert
            index_name: Name of the Pinecone index
            batch_size: Size of batches for upsertion
        """
        # Log input parameters
        mlflow.log_param("vector_store_type", "PineconeVectorStore")
        mlflow.log_param("index_name_for_upsertion", index_name)
        mlflow.log_param("num_text_chunks_for_upsertion", len(chunks))
        mlflow.log_param("batch_size for pinecone data injection", batch_size)
        if hasattr(self.embedding, "model"):
            mlflow.log_param("embedding_model_used_for_upsertion",
                           getattr(self.embedding, "model", "unknown"))

        print(f"Starting batched upsertion to Pinecone index '{index_name}' with {len(chunks)} chunks...")

        start_time = time.time()

        # Upsert in batches to avoid Pinecone 4MB limit
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=self.embedding,
                index_name=index_name
            )
            print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} chunks).")

        end_time = time.time()
        upsertion_duration = end_time - start_time

        # Log metrics
        mlflow.log_metric("pinecone_upsertion_duration_seconds", upsertion_duration)

        print(f"✅ Upsertion to Pinecone completed in {upsertion_duration:.2f} seconds.")

    def get_vector_store(self, index_name: str) -> PineconeVectorStore:
        """
        Get existing Pinecone vector store.

        Args:
            index_name: Name of the Pinecone index

        Returns:
            PineconeVectorStore object
        """
        return PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embedding
        )

    def test_embedding(self, test_text: str = "Hello to medical chatbot.") -> int:
        """
        Test the embedding model and return vector dimension.

        Args:
            test_text: Text to test embedding

        Returns:
            Dimension of the embedding vector
        """
        if not self.embedding:
            raise ValueError("Embedding model not set up. Call setup_embeddings first.")

        vector = self.embedding.embed_query(test_text)
        vector_dim = len(vector)
        print(f"Vector length: {vector_dim}")

        mlflow.log_param("vector_embedding_size", vector_dim)
        return vector_dim
