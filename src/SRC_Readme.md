## RAG pipeline and RAG evaluation using RAGAS

### Run the RAG evaluation pipeline - same as ragas.ipynb notebook but here , pipelines are seperated and made independent.

### Run

```
  python main_orchestrator.py
```

This will Run :

1. Data Loading Pipeline - PDF loading and text extraction

2. Data Processing Pipeline - Text splitting and filtering

3. Vector Database Pipeline - Embedding and Pinecone setup

4. RAG Pipeline - Retrieval and generation

5. Ground Truth Generation Pipeline - Question-answer generation

6. Evaluation Pipeline - RAGAS evaluation

#### All pipelines are independently placed in pipelines Folder under src Folder
