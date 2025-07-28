
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import mlflow  # Import MLflow for experiment tracking
import dagshub


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define LLM parameters
llm_model_name = "gemini-2.0-flash"
llm_temperature = 0
llm_max_tokens = None
llm_timeout = None
llm_max_retries = 2

chatModel = ChatGoogleGenerativeAI(
    model=llm_model_name,
    temperature=llm_temperature,
    max_tokens=llm_max_tokens,
    timeout=llm_timeout,
    max_retries=llm_max_retries,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    # Set up MLflow experiment tracking
    dagshub.init(repo_owner='264Gaurav', repo_name='medical-chatbot', mlflow=True)
    mlflow.set_experiment("PDF_Processing_Experiment")

    # Start an MLflow run to log this execution
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")


        # Log text splitter parameters
        chunk_size = 1000
        chunk_overlap = 50
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("text_splitter_class", "RecursiveCharacterTextSplitter")


        # Invoke the RAG chain and get the response
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])

        # Log the response and its length
        mlflow.log_param("response", response["answer"])
        mlflow.log_metric("response_length", len(response["answer"]))

        # Return the response
        return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)


















# from flask import Flask, render_template, jsonify, request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os


# app = Flask(__name__)


# load_dotenv()

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# embeddings = download_hugging_face_embeddings()

# index_name = "medical-chatbot"
# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )


# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# # Define LLM parameters
# llm_model_name = "gemini-2.0-flash"
# llm_temperature = 0
# llm_max_tokens = None
# llm_timeout = None
# llm_max_retries = 2


# chatModel = ChatGoogleGenerativeAI(
#     model=llm_model_name,
#     temperature=llm_temperature,
#     max_tokens=llm_max_tokens,
#     timeout=llm_timeout,
#     max_retries=llm_max_retries,
# )


# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# @app.route("/")
# def index():
#     return render_template('index.html')



# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])



# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8080, debug= True)
