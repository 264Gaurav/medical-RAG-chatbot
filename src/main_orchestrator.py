"""
Main Orchestrator for Medical RAG Chatbot.

This script demonstrates how to use all the modular pipelines together
to create a complete RAG system with evaluation.
"""

import os
import mlflow
import dagshub
from dotenv import load_dotenv

from pipelines.data_pipeline import DataPipeline
from pipelines.vector_pipeline import VectorPipeline
from pipelines.rag_pipeline import RAGPipeline
from pipelines.ground_truth_pipeline import GroundTruthPipeline
from pipelines.evaluation_pipeline import EvaluationPipeline


def setup_environment():
    """Set up environment variables and MLflow."""
    # Load environment variables
    load_dotenv()

    # Initialize DAGSHub and MLflow
    dagshub.init(repo_owner='264Gaurav', repo_name='medical-chatbot', mlflow=True)
    mlflow.set_experiment("RAG_ragas")

    print("Environment setup completed.")


def run_data_pipeline(data_dir: str = "data") -> tuple:
    """
    Run the data processing pipeline.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (extracted_data, chunks)
    """
    print("\n=== Running Data Pipeline ===")

    data_pipeline = DataPipeline(chunk_size=500, chunk_overlap=100)
    extracted_data, chunks = data_pipeline.process_data(data_dir)

    return extracted_data, chunks


def run_vector_pipeline(chunks, pinecone_api_key: str = None) -> tuple:
    """
    Run the vector database pipeline.

    Args:
        chunks: Document chunks
        pinecone_api_key: Pinecone API key

    Returns:
        Tuple of (vector_pipeline, vector_store)
    """
    print("\n=== Running Vector Pipeline ===")

    vector_pipeline = VectorPipeline(pinecone_api_key=pinecone_api_key)

    # Setup embeddings
    embeddings = vector_pipeline.setup_embeddings()
    vector_dim = vector_pipeline.test_embedding()

    # Setup Pinecone index
    index_name = 'medical-chatbot'
    vector_pipeline.setup_pinecone_index(index_name, vector_dim)

    # Get vector store
    vector_store = vector_pipeline.get_vector_store(index_name)

    # Uncomment the following line if you want to upsert documents
    # vector_pipeline.upsert_documents(chunks, index_name, batch_size=50)

    return vector_pipeline, vector_store


def run_rag_pipeline(vector_store) -> tuple:
    """
    Run the RAG pipeline.

    Args:
        vector_store: Pinecone vector store

    Returns:
        Tuple of (rag_pipeline, rag_chain, ragas_chain)
    """
    print("\n=== Running RAG Pipeline ===")

    rag_pipeline = RAGPipeline()

    # Setup LLM
    rag_pipeline.setup_llm(model_name="llama3.1:latest", temperature=0)

    # Setup prompt
    rag_pipeline.setup_prompt(use_hub_prompt=True)

    # Setup retriever
    rag_pipeline.setup_retriever(vector_store, search_type="similarity", k=8)

    # Create RAG chains
    rag_chain, ragas_chain = rag_pipeline.create_rag_chains()

    # Test retrieval
    retrieved_docs, retrieval_duration, num_retrieved = rag_pipeline.test_retrieval()

    # Test RAG chain
    test_question = "What dose of vitamin D should we take? What diseases can occur due to a lack of vitamin D?"
    response = rag_pipeline.query_rag_chain(test_question)
    print(f"\nTest Question: {test_question}")
    print(f"Response: {response}")

    return rag_pipeline, rag_chain, ragas_chain


def run_ground_truth_pipeline(chunks, num_chunks: int = 30) -> tuple:
    """
    Run the ground truth generation pipeline.

    Args:
        chunks: Document chunks
        num_chunks: Number of chunks to process

    Returns:
        Tuple of (ground_truth_pipeline, eval_dataset)
    """
    print("\n=== Running Ground Truth Pipeline ===")

    ground_truth_pipeline = GroundTruthPipeline()

    # Setup LLMs
    ground_truth_pipeline.setup_question_generation_llm(model_name="llama3.1:latest", temperature=0.5)
    ground_truth_pipeline.setup_answer_generation_llm(model_name="llama3.1:latest", temperature=0.5)

    # Setup parsers
    ground_truth_pipeline.setup_parsers()

    # Generate ground truth dataset
    eval_dataset = ground_truth_pipeline.generate_ground_truth_dataset(
        chunks, num_chunks, "groundtruth_eval_dataset.csv"
    )

    return ground_truth_pipeline, eval_dataset


def run_evaluation_pipeline(rag_pipeline, eval_dataset, embeddings) -> dict:
    """
    Run the evaluation pipeline.

    Args:
        rag_pipeline: RAG pipeline with retrieval_augmented_qa_chain
        eval_dataset: Ground truth evaluation dataset
        embeddings: Embeddings for evaluation

    Returns:
        Evaluation results
    """
    print("\n=== Running Evaluation Pipeline ===")

    evaluation_pipeline = EvaluationPipeline()

    # Setup evaluation components
    evaluation_pipeline.setup_evaluation_llm(model_name="llama3.1:latest", temperature=0.0)
    evaluation_pipeline.setup_embeddings(embeddings)

    # Run complete evaluation
    results = evaluation_pipeline.run_complete_evaluation(
        rag_pipeline.retrieval_augmented_qa_chain,
        eval_dataset,
        "basic_qa_ragas_dataset.csv"
    )

    return results


def main():
    """Main function to orchestrate all pipelines."""
    print("🚀 Starting Medical RAG Chatbot Pipeline Orchestration")

    try:
        # Setup environment
        setup_environment()

        # Run data pipeline
        extracted_data, chunks = run_data_pipeline()

        # Run vector pipeline
        vector_pipeline, vector_store = run_vector_pipeline(chunks)

        # Run RAG pipeline
        rag_pipeline, rag_chain, ragas_chain = run_rag_pipeline(vector_store)

        # Run ground truth pipeline
        ground_truth_pipeline, eval_dataset = run_ground_truth_pipeline(chunks)

        # Run evaluation pipeline
        evaluation_results = run_evaluation_pipeline(
            rag_pipeline,
            eval_dataset,
            vector_pipeline.embedding
        )

        print("\n🎉 All pipelines completed successfully!")
        print("\n=== Summary ===")
        print(f"Documents processed: {len(extracted_data)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        print(f"Evaluation results logged to MLflow")

    except Exception as e:
        print(f"❌ Error in pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()
