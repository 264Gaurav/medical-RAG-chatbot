"""
Ground Truth Pipeline for medical RAG chatbot.

Handles:
- Question generation from context
- Answer generation for questions
- Dataset creation for evaluation
"""

import pandas as pd
import json
from tqdm import tqdm
from datasets import Dataset
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import List, Dict, Any
import mlflow


class GroundTruthPipeline:
    """Pipeline for generating ground truth datasets."""

    def __init__(self):
        """Initialize the ground truth pipeline."""
        self.question_generation_llm = None
        self.answer_generation_llm = None
        self.question_output_parser = None
        self.answer_output_parser = None

    def setup_question_generation_llm(self, model_name: str = "llama3.1:latest",
                                    temperature: float = 0.5) -> OllamaLLM:
        """
        Set up the LLM for question generation.

        Args:
            model_name: Name of the Ollama model
            temperature: Temperature for generation

        Returns:
            Configured Ollama model
        """
        print(f"Initializing Ollama model for question generation: {model_name}...")

        question_generation_llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
        )

        self.question_generation_llm = question_generation_llm
        print(f"Question generation LLM '{model_name}' initialized.")
        return question_generation_llm

    def setup_answer_generation_llm(self, model_name: str = "llama3.1:latest",
                                  temperature: float = 0.5) -> OllamaLLM:
        """
        Set up the LLM for answer generation.

        Args:
            model_name: Name of the Ollama model
            temperature: Temperature for generation

        Returns:
            Configured Ollama model
        """
        print(f"Initializing Ollama model for answer generation: {model_name}...")

        answer_generation_llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
        )

        self.answer_generation_llm = answer_generation_llm
        print(f"Answer generation LLM '{model_name}' initialized.")
        return answer_generation_llm

    def setup_parsers(self) -> None:
        """Set up output parsers for structured generation."""
        # Question schema
        question_schema = ResponseSchema(
            name="question",
            description="a question about the context."
        )
        question_response_schemas = [question_schema]
        self.question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)

        # Answer schema
        answer_schema = ResponseSchema(
            name="answer",
            description="an answer to the question"
        )
        answer_response_schemas = [answer_schema]
        self.answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)

        print("Output parsers configured for question and answer generation.")

    def generate_questions(self, chunks: List[Document], num_chunks: int = 30, start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Generate questions from text chunks.

        Args:
            chunks: List of Document objects
            num_chunks: Number of chunks to process

        Returns:
            List of question-context pairs
        """
        if not self.question_generation_llm or not self.question_output_parser:
            raise ValueError("Question generation LLM and parser must be set up first")

        # Create question generation prompt
        qa_template = """\
You are a Professor of Medical University creating a test for advanced students.
For each context, create a question that is specific to the context. Avoid generic questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

        prompt_template = ChatPromptTemplate.from_template(template=qa_template)
        bare_prompt_template = "{content}"
        bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)
        question_generation_chain = bare_template | self.question_generation_llm

        qac_triples = []

        print(f"Generating questions for {num_chunks} chunks starting at index {start_index}...")
        end_index = start_index + num_chunks
        for text in tqdm(chunks[start_index:end_index]):
            messages = prompt_template.format_messages(
                context=text,
                format_instructions=self.question_output_parser.get_format_instructions()
            )
            response = question_generation_chain.invoke({"content": messages})
            try:
                output_dict = self.question_output_parser.parse(response)
                output_dict["context"] = text
                qac_triples.append(output_dict)
            except Exception as e:
                continue

        print(f"Generated {len(qac_triples)} questions successfully.")
        return qac_triples

    def generate_answers(self, qac_triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate answers for the generated questions.

        Args:
            qac_triples: List of question-context pairs

        Returns:
            List of question-answer-context triples
        """
        if not self.answer_generation_llm or not self.answer_output_parser:
            raise ValueError("Answer generation LLM and parser must be set up first")

        # Create answer generation prompt
        qa_template = """\
You are a Professor of Medical university creating a test for advanced students.
For each question and context, create an answer.

answer: a answer about the context.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""

        prompt_template = ChatPromptTemplate.from_template(template=qa_template)
        bare_prompt_template = "{content}"
        bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)
        answer_generation_chain = bare_template | self.answer_generation_llm

        print("Generating answers for questions...")
        for triple in tqdm(qac_triples):
            messages = prompt_template.format_messages(
                context=triple["context"],
                question=triple["question"],
                format_instructions=self.answer_output_parser.get_format_instructions()
            )
            response = answer_generation_chain.invoke({"content": messages})
            try:
                output_dict = self.answer_output_parser.parse(response)
                triple["answer"] = output_dict["answer"]
            except Exception as e:
                continue

        print(f"Generated answers for {len(qac_triples)} questions.")
        return qac_triples

    def create_evaluation_dataset(self, qac_triples: List[Dict[str, Any]]) -> Dataset:
        """
        Create evaluation dataset from question-answer-context triples.

        Args:
            qac_triples: List of question-answer-context triples

        Returns:
            HuggingFace Dataset for evaluation
        """
        # Convert to DataFrame and clean
        ground_truth_qac_set = pd.DataFrame(qac_triples)
        ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x))
        ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer": "ground_truth"})

        # Drop irrelevant columns if they exist
        columns_to_drop = ["additional_kwargs", "response_metadata"]
        for col in columns_to_drop:
            if col in ground_truth_qac_set.columns:
                ground_truth_qac_set = ground_truth_qac_set.drop(columns=[col])

        # Convert to HuggingFace Dataset
        print("ground_truth_qac_set :",ground_truth_qac_set)
        eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

        print(f"Created evaluation dataset with {len(eval_dataset)} samples.")
        return eval_dataset

    def save_dataset(self, dataset: Dataset, filename: str) -> None:
        """
        Save dataset to CSV file.

        Args:
            dataset: HuggingFace Dataset to save
            filename: Output filename
        """
        dataset.to_csv(filename)
        print(f"Dataset saved to {filename}")

    def generate_ground_truth_dataset(self, chunks: List[Document],
                                    num_chunks: int = 30,
                                    output_filename: str = "groundtruth_eval_dataset.csv",
                                    start_index: int = 0) -> Dataset:
        """
        Complete pipeline for generating ground truth dataset.

        Args:
            chunks: List of Document objects
            num_chunks: Number of chunks to process
            output_filename: Output CSV filename

        Returns:
            Generated evaluation dataset
        """
        print("Starting ground truth dataset generation...")

        # Generate questions
        qac_triples = self.generate_questions(chunks, num_chunks, start_index)

        # Generate answers
        qac_triples = self.generate_answers(qac_triples)

        # Create dataset
        eval_dataset = self.create_evaluation_dataset(qac_triples)

        # Save dataset
        self.save_dataset(eval_dataset, output_filename)

        print("Ground truth dataset generation completed successfully.")
        return eval_dataset
