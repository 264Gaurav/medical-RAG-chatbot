"""
Data Pipeline for medical RAG chatbot.

Handles:
- PDF document loading
- Text splitting into chunks
- Document filtering and cleaning
"""

import os
import time
import json
import mlflow
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DataPipeline:
    """Pipeline for loading and processing PDF documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the data pipeline.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_pdf_files(self, data_dir: str) -> List[Document]:
        """
        Load PDF files from a directory.

        Args:
            data_dir: Path to directory containing PDF files

        Returns:
            List of loaded Document objects
        """
        loader = DirectoryLoader(
            data_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()

    def filter_docs(self, docs: List[Document]) -> List[Document]:
        """
        Filter documents to keep only essential metadata.

        Args:
            docs: List of Document objects to filter

        Returns:
            List of filtered Document objects with minimal metadata
        """
        # Log initial state
        input_documents_count = len(docs)
        mlflow.log_metric("input_documents_count_for_filtering", input_documents_count)
        mlflow.log_param("filter_function_name", "filter_docs")

        minimal_docs: List[Document] = []

        for doc in docs:
            src = doc.metadata.get('source')
            page = doc.metadata.get('page')
            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": src, "page": page}
                )
            )

        output_documents_count = len(minimal_docs)
        documents_filtered_count = input_documents_count - output_documents_count

        # Log metrics after processing
        mlflow.log_metric("output_documents_count_after_filtering", output_documents_count)
        mlflow.log_metric("documents_filtered_count", documents_filtered_count)

        # Log sample of filtered documents as artifact
        if minimal_docs:
            sample_docs_path = "filtered_docs_sample.json"
            sample_data = []
            for i, doc in enumerate(minimal_docs[:5]):
                sample_data.append({
                    "page_content_preview": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            with open(sample_docs_path, "w") as f:
                json.dump(sample_data, f, indent=4)
            mlflow.log_artifact(sample_docs_path, artifact_path="data_filtering_artifacts")
            os.remove(sample_docs_path)

        return minimal_docs

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            docs: List of Document objects to split

        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(docs)

        # Log metrics
        num_chunks = len(chunks)
        mlflow.log_metric("num_chunks_created", num_chunks)
        mlflow.log_param("chunk_size", self.chunk_size)
        mlflow.log_param("chunk_overlap", self.chunk_overlap)
        mlflow.log_param("text_splitter_class", "RecursiveCharacterTextSplitter")

        # Log sample chunks as artifact
        if chunks:
            chunks_sample_path = "chunks_sample.json"
            with open(chunks_sample_path, "w") as f:
                json.dump([chunk.model_dump() for chunk in chunks[:5]], f, indent=4)
            mlflow.log_artifact(chunks_sample_path, artifact_path="data_processing_artifacts")
            os.remove(chunks_sample_path)

        return chunks

    def process_data(self, data_dir: str) -> tuple[List[Document], List[Document]]:
        """
        Complete data processing pipeline.

        Args:
            data_dir: Path to directory containing PDF files

        Returns:
            Tuple of (extracted_docs, chunks)
        """
        # Start MLflow run
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID: {run_id}")

            mlflow.log_param("data_directory", data_dir)
            mlflow.log_param("pdf_loader_class", "PyPDFLoader")

            # Load PDFs
            start_time = time.time()
            extracted_data = self.load_pdf_files(data_dir)
            end_time = time.time()
            loading_duration = end_time - start_time

            mlflow.log_metric("num_documents_loaded", len(extracted_data))
            mlflow.log_metric("pdf_loading_duration_seconds", loading_duration)
            mlflow.log_param("num_documents", len(extracted_data))

            print(f"Loaded {len(extracted_data)} documents in {loading_duration:.2f} seconds.")

            # Filter documents
            minimal_docs = self.filter_docs(extracted_data)

            # Split into chunks
            chunks = self.split_documents(minimal_docs)

            print(f"Created {len(chunks)} text chunks.")
            print(f"MLflow run finished. View at {mlflow.get_tracking_uri()}")

            return extracted_data, chunks
