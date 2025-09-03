"""
RAG Pipeline for medical RAG chatbot.

Handles:
- LLM model setup
- RAG chain creation
- Retrieval operations
"""

import time
import mlflow
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from operator import itemgetter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings


class RAGPipeline:
    """Pipeline for RAG operations."""

    def __init__(self):
        """Initialize the RAG pipeline."""
        self.chat_model = None
        self.retriever = None
        self.rag_chain = None
        self.retrieval_augmented_qa_chain = None
        self.prompt = None

    def setup_llm(self, model_name: str = "llama3.1:latest",
                  temperature: float = 0, max_retries: int = 2) -> OllamaLLM:
        """
        Set up the LLM model.

        Args:
            model_name: Name of the Ollama model
            temperature: Temperature for generation
            max_retries: Maximum retry attempts

        Returns:
            Configured Ollama model
        """
        # Log LLM parameters
        mlflow.log_param("llm_provider", "Ollama")
        mlflow.log_param("llm_class", "Ollama")
        mlflow.log_param("llm_model_name", model_name)
        mlflow.log_param("llm_temperature", temperature)
        mlflow.log_param("llm_max_retries", max_retries)

        print(f"Initializing Ollama model: {model_name}...")
        start_time = time.time()

        chat_model = OllamaLLM(
            model=model_name,
            temperature=temperature,
        )

        end_time = time.time()
        initialization_duration = end_time - start_time

        # Log metrics
        mlflow.log_metric("llm_initialization_duration_seconds", initialization_duration)

        print(f"Ollama model '{model_name}' initialized in {initialization_duration:.4f} seconds.")

        self.chat_model = chat_model
        return chat_model

    def setup_prompt(self, use_hub_prompt: bool = True, custom_system_prompt: str = None) -> ChatPromptTemplate:
        """
        Set up the prompt template.

        Args:
            use_hub_prompt: Whether to use LangChain Hub prompt
            custom_system_prompt: Custom system prompt if not using hub

        Returns:
            Configured prompt template
        """
        if use_hub_prompt:
            # Pull a pre-made RAG prompt from LangChain Hub
            prompt = hub.pull("rlm/rag-prompt")
            mlflow.log_param("prompt_source", "LangChain Hub (rlm/rag-prompt)")
        else:
            # Use custom system prompt
            system_prompt = custom_system_prompt or (
                "You are an Medical assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Keep the answer concise and understandable."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            mlflow.log_param("prompt_source", "custom")
            mlflow.log_param("system_prompt_for_llm", system_prompt)

        self.prompt = prompt
        print(f"Prompt template defined and logged to MLflow.")
        return prompt

    def setup_retriever(self, vector_store: PineconeVectorStore,
                       search_type: str = "similarity", k: int = 8) -> None:
        """
        Set up the retriever.

        Args:
            vector_store: Pinecone vector store
            search_type: Type of search (similarity, mmr, etc.)
            k: Number of documents to retrieve
        """
        # Log retrieval parameters
        mlflow.log_param("retriever_search_type", search_type)
        mlflow.log_param("retriever_k_value", k)

        # Configure the retriever
        self.retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        print(f"Retriever configured with search_type='{search_type}' and k={k}")

    def create_rag_chains(self) -> tuple:
        """
        Create RAG chains.

        Returns:
            Tuple of (rag_chain, retrieval_augmented_qa_chain)
        """
        if not self.retriever or not self.prompt or not self.chat_model:
            raise ValueError("Retriever, prompt, and chat model must be set up first")

        # Helper function to format retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Define the full RAG chain
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.chat_model
            | StrOutputParser()
        )

        # RAGAS compatible chain
        retrieval_augmented_qa_chain = (
            {"context": itemgetter("question") | self.retriever, "question": itemgetter("question")}
            | RunnablePassthrough.assign(context=itemgetter("context"))
            | {"response": self.prompt | self.chat_model, "context": itemgetter("context")}
        )

        self.rag_chain = rag_chain
        self.retrieval_augmented_qa_chain = retrieval_augmented_qa_chain

        print("RAG chains created successfully.")
        return rag_chain, retrieval_augmented_qa_chain

    def test_retrieval(self, test_query: str = "What is Cancer?") -> tuple:
        """
        Test the retrieval system.

        Args:
            test_query: Query to test retrieval

        Returns:
            Tuple of (retrieved_docs, retrieval_duration, num_retrieved)
        """
        if not self.retriever:
            raise ValueError("Retriever not set up. Call setup_retriever first.")

        # Log the specific query used
        mlflow.log_param("retrieval_query", test_query)

        # Measure and log retrieval time
        start_time = time.time()
        retrieved_docs = self.retriever.invoke(test_query)
        end_time = time.time()
        retrieval_duration = end_time - start_time

        mlflow.log_metric("retrieval_duration_seconds", retrieval_duration)
        num_retrieved = len(retrieved_docs)
        mlflow.log_metric("num_retrieved_documents", num_retrieved)

        print(f"Retrieved {num_retrieved} documents for '{test_query}' in {retrieval_duration:.4f} seconds.")

        return retrieved_docs, retrieval_duration, num_retrieved

    def query_rag_chain(self, question: str) -> str:
        """
        Query the RAG chain.

        Args:
            question: Question to ask

        Returns:
            Generated answer
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not created. Call create_rag_chains first.")

        response = self.rag_chain.invoke(question)
        return response

    def query_ragas_chain(self, question: str) -> dict:
        """
        Query the RAGAS compatible chain.

        Args:
            question: Question to ask

        Returns:
            Dictionary with response and context
        """
        if not self.retrieval_augmented_qa_chain:
            raise ValueError("RAGAS chain not created. Call create_rag_chains first.")

        response = self.retrieval_augmented_qa_chain.invoke({"question": question})
        return response
