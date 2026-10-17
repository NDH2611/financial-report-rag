# Financial Report Analysis using LLM + RAG

MVP for financial report analysis using LLM and Retrieval-Augmented Generation.

## Pipeline

Document Processing
-> Text/Table Extraction
-> Chunking
-> Embedding
-> FAISS
-> Retrieval
-> LLM
-> Evaluation

## Dataset

Apple Inc. Form 10-K, Fiscal Year 2025.

## Team Modules

- Data Extraction: `src/extraction`
- RAG Retrieval: `src/retrieval`
- LLM Generation: `src/generation`
- Evaluation: `src/evaluation`
- Demo UI: `app.py`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py