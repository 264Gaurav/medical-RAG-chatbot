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

repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "264Gaurav")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "medical-chatbot")
exp_name = os.getenv("MLFLOW_EXPERIMENT", "Ragas")
CHUNK_SIZE = os.getenv("CHUNK_SIZE",500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP",100)
PINECONE_INDEX = os.getenv("PINECONE_INDEX",'medical-chatbox')
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE",0.0)
LLM_MODEL = os.getenv("LLM_MODEL",'llama3.1:latest')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",'nomic-embed-text:latest')
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",'http://localhost:11434')
RETRIEVER_SEARCH_TYPE = os.getenv("RETRIEVER_SEARCH_TYPE",'similarity')
TOP_K = os.getenv("TOP_K",3)
TEST_QUESTION = os.getenv("TEST_QUESTION",'What dose of vitamin-C and vitamin-D should we take?')
NUM_QA_CHUNKS = os.getenv("NUM_QA_CHUNKS",30) #No. of Qus for evaluation
QUESTION_LLM_MODEL = os.getenv("QUESTION_LLM_MODEL",'llama3.1:latest')
ANSWER_LLM_MODEL = os.getenv("ANSWER_LLM_MODEL",'llama3.1:latest')
QUESTION_LLM_TEMPERATURE = os.getenv("QUESTION_LLM_TEMPERATURE",0.5)
ANSWER_LLM_TEMPERATURE = os.getenv("ANSWER_LLM_TEMPERATURE",0.5)
GROUND_TRUTH_FILENAME = os.getenv("GROUND_TRUTH_FILENAME",'groundtruth_eval_dataset.csv')
RAGAS_DATASET_FILENAME = os.getenv("RAGAS_DATASET_FILENAME",'basic_qa_ragas_dataset.csv')
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL",'llama3.1:latest')
EVAL_LLM_TEMPERATURE = os.getenv("EVAL_LLM_TEMPERATURE",0.0)

def setup_environment():
    """Set up environment variables and MLflow."""
    # Load environment variables
    load_dotenv()

    # Initialize DAGSHub and MLflow
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_experiment(exp_name)

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

    data_pipeline = DataPipeline(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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
    rag_pipeline.setup_llm(model_name=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # Setup prompt
    rag_pipeline.setup_prompt(use_hub_prompt=True)

    # Setup retriever
    rag_pipeline.setup_retriever(vector_store, search_type=RETRIEVER_SEARCH_TYPE, k=TOP_K)

    # Create RAG chains
    rag_chain, ragas_chain = rag_pipeline.create_rag_chains()

    # Test retrieval
    retrieved_docs, retrieval_duration, num_retrieved = rag_pipeline.test_retrieval()

    # Test RAG chain
    test_question = TEST_QUESTION
    response = rag_pipeline.query_rag_chain(test_question)
    print(f"\nTest Question: {test_question}")
    print(f"Response: {response}")

    return rag_pipeline, rag_chain, ragas_chain


def run_ground_truth_pipeline(chunks, num_chunks: int = NUM_QA_CHUNKS) -> tuple:
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
    ground_truth_pipeline.setup_question_generation_llm(model_name=QUESTION_LLM_MODEL, temperature=QUESTION_LLM_TEMPERATURE)
    ground_truth_pipeline.setup_answer_generation_llm(model_name=ANSWER_LLM_MODEL, temperature=ANSWER_LLM_TEMPERATURE)

    # Setup parsers
    ground_truth_pipeline.setup_parsers()

    # Generate ground truth dataset
    eval_dataset = ground_truth_pipeline.generate_ground_truth_dataset(
        chunks, num_chunks, GROUND_TRUTH_FILENAME
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
    evaluation_pipeline.setup_evaluation_llm(model_name=EVAL_LLM_MODEL, temperature=EVAL_LLM_TEMPERATURE)
    evaluation_pipeline.setup_embeddings(embeddings)

    # Run complete evaluation
    results = evaluation_pipeline.run_complete_evaluation(
        rag_pipeline.retrieval_augmented_qa_chain,
        eval_dataset,
        RAGAS_DATASET_FILENAME
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

        ## Run ground truth pipeline
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
