from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
import mlflow  # Import MLflow for experiment tracking


# Extract Data From the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)

    documents = loader.load()

    # Log the number of documents loaded
    with mlflow.start_run():
        mlflow.log_param("pdf_directory", data)
        mlflow.log_metric("num_documents_loaded", len(documents))

    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    # Log the number of filtered documents
    with mlflow.start_run():
        mlflow.log_metric("num_filtered_documents", len(minimal_docs))

    return minimal_docs


# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)

    # Log the number of text chunks created
    with mlflow.start_run():
        mlflow.log_param("chunk_size", 1000)
        mlflow.log_param("chunk_overlap", 50)
        mlflow.log_metric("num_text_chunks", len(text_chunks))

    return text_chunks


# Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)  # this model returns 384 dimensions

    # Log the embedding model used
    with mlflow.start_run():
        mlflow.log_param("embedding_model", model_name)

    return embeddings
















# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from typing import List
# from langchain.schema import Document


# #Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)

#     documents=loader.load()

#     return documents



# def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
#     """
#     Given a list of Document objects, return a new list of Document objects
#     containing only 'source' in metadata and the original page_content.
#     """
#     minimal_docs: List[Document] = []
#     for doc in docs:
#         src = doc.metadata.get("source")
#         minimal_docs.append(
#             Document(
#                 page_content=doc.page_content,
#                 metadata={"source": src}
#             )
#         )
#     return minimal_docs



# #Split the Data into Text Chunks
# def text_split(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks



# #Download the Embeddings from HuggingFace
# def download_hugging_face_embeddings():
#     embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
#     return embeddings
