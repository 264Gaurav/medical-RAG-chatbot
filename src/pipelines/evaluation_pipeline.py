"""
Evaluation Pipeline for medical RAG chatbot.

Handles:
- RAGAS evaluation setup
- Evaluation metrics calculation
- Results logging and analysis
"""

import os
import json
import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness,
    answer_similarity,
)
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import Dict, Any, List


class EvaluationPipeline:
    """Pipeline for RAGAS evaluation."""

    def __init__(self):
        """Initialize the evaluation pipeline."""
        self.evaluation_llm = None
        self.embeddings = None

    def setup_evaluation_llm(self, model_name: str = "llama3.1:latest",
                           temperature: float = 0.0) -> OllamaLLM:
        """
        Set up the LLM for evaluation.

        Args:
            model_name: Name of the Ollama model
            temperature: Temperature for generation (keep deterministic for evaluation)

        Returns:
            Configured Ollama model
        """
        print(f"Initializing Ollama model for evaluation: {model_name}...")

        evaluation_llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
        )

        # Log parameters
        mlflow.log_param("evaluation llm_provider", "Ollama")
        mlflow.log_param("evaluation llm_model_name", model_name)
        mlflow.log_param("evaluation llm_temperature", temperature)
        mlflow.log_param("evaluation llm_max_retries", 2)

        self.evaluation_llm = evaluation_llm
        print(f"Evaluation LLM '{model_name}' initialized.")
        return evaluation_llm

    def setup_embeddings(self, embeddings: OllamaEmbeddings) -> None:
        """
        Set up embeddings for evaluation.

        Args:
            embeddings: OllamaEmbeddings object
        """
        self.embeddings = embeddings
        print("Embeddings configured for evaluation.")

    def create_ragas_dataset(self, rag_pipeline, eval_dataset: Dataset) -> Dataset:
        """
        Create RAGAS dataset from RAG pipeline and evaluation dataset.

        Args:
            rag_pipeline: RAG pipeline with retrieval_augmented_qa_chain
            eval_dataset: Ground truth evaluation dataset

        Returns:
            RAGAS-compatible dataset
        """
        rag_dataset = []

        print("Creating RAGAS dataset...")
        for row in tqdm(eval_dataset):
            try:
                answer = rag_pipeline.invoke({"question": row["question"]})
                rag_dataset.append({
                    "question": row["question"],
                    "answer": answer["response"],
                    "contexts": [context.page_content for context in answer["context"]],
                    "reference": row["ground_truth"]
                })
            except Exception as e:
                print(f"Error processing question: {e}")
                continue

        rag_df = pd.DataFrame(rag_dataset)
        rag_eval_dataset = Dataset.from_pandas(rag_df)

        print(f"Created RAGAS dataset with {len(rag_eval_dataset)} samples.")
        return rag_eval_dataset

    def save_ragas_dataset(self, dataset: Dataset, filename: str) -> None:
        """
        Save RAGAS dataset to CSV file.

        Args:
            dataset: RAGAS dataset to save
            filename: Output filename
        """
        dataset.to_csv(filename)
        mlflow.log_param("Data set file for RAG evaluation", filename)
        print(f"RAGAS dataset saved to {filename}")

    def evaluate_retrieval_metrics(self, ragas_dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate retrieval-related metrics (context precision, context recall).

        Args:
            ragas_dataset: RAGAS dataset for evaluation

        Returns:
            Dictionary containing evaluation results
        """
        if not self.evaluation_llm or not self.embeddings:
            raise ValueError("Evaluation LLM and embeddings must be set up first")

        print("Evaluating retrieval metrics (context precision, context recall)...")

        result = evaluate(
            ragas_dataset,
            metrics=[
                context_precision,
                context_recall
            ],
            llm=self.evaluation_llm,
            embeddings=self.embeddings,
        )

        # Log metrics
        context_precision_values = result['context_precision']
        context_recall_values = result['context_recall']

        mlflow.log_metric("context_precision_mean", np.nanmean(context_precision_values))
        mlflow.log_metric("context_precision_min", np.nanmin(context_precision_values))
        mlflow.log_metric("context_precision_max", np.nanmax(context_precision_values))

        mlflow.log_metric("context_recall_mean", np.nanmean(context_recall_values))
        mlflow.log_metric("context_recall_min", np.nanmin(context_recall_values))
        mlflow.log_metric("context_recall_max", np.nanmax(context_recall_values))

        print("Retrieval metrics evaluation completed.")
        return result

    def evaluate_generation_metrics(self, ragas_dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate generation-related metrics (faithfulness, answer relevancy, etc.).

        Args:
            ragas_dataset: RAGAS dataset for evaluation

        Returns:
            Dictionary containing evaluation results
        """
        if not self.evaluation_llm or not self.embeddings:
            raise ValueError("Evaluation LLM and embeddings must be set up first")

        print("Evaluating generation metrics (faithfulness, answer relevancy, etc.)...")

        result = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_correctness,
                answer_similarity,
            ],
            llm=self.evaluation_llm,
            embeddings=self.embeddings,
        )

        # Log metrics
        faithfulness_values = result['faithfulness']
        answer_relevancy_values = result['answer_relevancy']
        answer_correctness_values = result['answer_correctness']
        answer_similarity_values = result['answer_similarity']

        # Handle NaN values safely
        if len(faithfulness_values) > 0 and not np.all(np.isnan(faithfulness_values)):
            mlflow.log_metric("faithfulness_mean", float(np.nanmean(faithfulness_values)))
            mlflow.log_metric("faithfulness_min", float(np.nanmin(faithfulness_values)))
            mlflow.log_metric("faithfulness_max", float(np.nanmax(faithfulness_values)))

        if len(answer_relevancy_values) > 0 and not np.all(np.isnan(answer_relevancy_values)):
            mlflow.log_metric("answer_relevancy_mean", float(np.nanmean(answer_relevancy_values)))
            mlflow.log_metric("answer_relevancy_min", float(np.nanmin(answer_relevancy_values)))
            mlflow.log_metric("answer_relevancy_max", float(np.nanmax(answer_relevancy_values)))

        if len(answer_correctness_values) > 0 and not np.all(np.isnan(answer_correctness_values)):
            mlflow.log_metric("answer_correctness_mean", float(np.nanmean(answer_correctness_values)))
            mlflow.log_metric("answer_correctness_min", float(np.nanmin(answer_correctness_values)))
            mlflow.log_metric("answer_correctness_max", float(np.nanmax(answer_correctness_values)))

        if len(answer_similarity_values) > 0 and not np.all(np.isnan(answer_similarity_values)):
            mlflow.log_metric("answer_similarity_mean", float(np.nanmean(answer_similarity_values)))
            mlflow.log_metric("answer_similarity_min", float(np.nanmin(answer_similarity_values)))
            mlflow.log_metric("answer_similarity_max", float(np.nanmax(answer_similarity_values)))

        print("Generation metrics evaluation completed.")
        return result

    def log_sample_results(self, ragas_dataset: Dataset, sample_index: int = 0) -> None:
        """
        Log a sample from the RAGAS dataset as an artifact.

        Args:
            ragas_dataset: RAGAS dataset
            sample_index: Index of the sample to log
        """
        if len(ragas_dataset) > sample_index:
            example_qa = ragas_dataset[sample_index]

            sample_file_path = "basic_qa_ragas_dataset_sample.json"
            with open(sample_file_path, "w") as f:
                json.dump(example_qa, f, indent=4)

            mlflow.log_artifact(sample_file_path, artifact_path="basic_qa_ragas_dataset_sample")
            os.remove(sample_file_path)

            print(f"Logged sample result at index {sample_index}")

    def run_complete_evaluation(self, rag_pipeline, eval_dataset: Dataset,
                              output_filename: str = "basic_qa_ragas_dataset.csv") -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Args:
            rag_pipeline: RAG pipeline with retrieval_augmented_qa_chain
            eval_dataset: Ground truth evaluation dataset
            output_filename: Output filename for RAGAS dataset

        Returns:
            Dictionary containing all evaluation results
        """
        print("Starting complete evaluation pipeline...")

        # Create RAGAS dataset
        ragas_dataset = self.create_ragas_dataset(rag_pipeline, eval_dataset)

        # Save RAGAS dataset
        self.save_ragas_dataset(ragas_dataset, output_filename)

        # Log sample results
        self.log_sample_results(ragas_dataset)

        # Evaluate retrieval metrics
        retrieval_results = self.evaluate_retrieval_metrics(ragas_dataset)

        # Evaluate generation metrics
        generation_results = self.evaluate_generation_metrics(ragas_dataset)

        # Combine results
        all_results = {
            "retrieval": retrieval_results,
            "generation": generation_results,
            "ragas_dataset": ragas_dataset
        }

        print("Complete evaluation pipeline finished successfully.")
        return all_results
