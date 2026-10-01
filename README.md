# PDF Knowledge Assistant with Neo4j and Gemini

A modular Python application that lets users upload PDF documents, ask questions about their contents, and inspect the passages retrieved to support an AI-generated answer.

The application combines **Docling document extraction**, **Gemini embeddings**, **Neo4j storage and retrieval**, and a **Gradio web interface**. It demonstrates retrieval-augmented generation (RAG): locating relevant document text before asking a language model to answer a question.

## Purpose

Research reports, technical documents, and institutional resources often contain information that is difficult to locate quickly. This project provides an interactive way to search those documents and generate an answer from relevant passages.

Users can upload a PDF, view indexing statistics, ask questions, inspect supporting sources, and manage indexed documents. The system is a prototype intended for development and learning; it has no reported answer-quality benchmark or production deployment validation.

## Technologies

| Technology | Role |
|---|---|
| Python | Application logic and coordination of the RAG pipeline |
| Gradio | Browser interface for PDF uploads, questions, and database management |
| Docling | Extracting PDF content and exporting it to Markdown |
| Google Gemini | Generating document/query embeddings and context-based answers |
| `google-generativeai` | Gemini SDK used in the supplied implementation |
| Neo4j Python driver and Cypher | Storing document chunks, retrieving passages, and managing records |
| python-dotenv | Loading credentials and connection settings from `.env` |

NumPy is listed in `requirements.txt`, although it is not directly used by the supplied source files.

## Workflow

### Index a document

1. The user uploads a PDF through Gradio.
2. Docling extracts content with OCR and table-structure processing disabled.
3. Markdown text is split into paragraphs. Paragraphs longer than the configured minimum length become chunks.
4. Gemini creates a document embedding for each chunk.
5. Each chunk is stored as a Neo4j `Chunk` node containing its text, embedding, source filename, chunk index, and metadata.
6. The interface displays processing statistics and database status.

### Answer a question

1. Gemini converts the question into a query embedding.
2. A Cypher query ranks stored chunks by the dot product of their embeddings with the query embedding.
3. The highest-ranked passages are included in the generation prompt.
4. Gemini is instructed to answer using the supplied context and acknowledge insufficient information.
5. Gradio displays the answer alongside source filenames, chunk indices, retrieval scores, and text previews.

These source passages help users review an answer. They do not automatically verify every statement made by the model.

## Features Implemented in the Source

- PDF upload and document processing.
- Separate modules for extraction, embeddings, storage, generation, and orchestration.
- Persistent chunk storage in Neo4j.
- Configurable retrieval count, with a UI range of 1–10 passages.
- Source previews alongside answers.
- Indexed-document lists and chunk counts.
- Document deletion by source filename.
- Retry handling for API errors indicating rate limits or quota limits.
- Environment-based configuration and required-credential validation.

## Repository Files

Save the uploaded files with the names below, removing the `(1)` suffix. The Python imports expect these exact module names.

| File | Responsibility |
|---|---|
| `app.py` | Gradio interface and event handlers |
| `config.py` | Environment variables, model identifiers, and default settings |
| `pdf_processor.py` | Docling extraction, paragraph chunks, and statistics |
| `embeddings.py` | Gemini document and query embeddings |
| `vector_store.py` | Neo4j schema, storage, retrieval, and deletion |
| `response_generator.py` | Context formatting, prompts, and Gemini answers |
| `rag_orchestrator.py` | Coordination of the indexing and question-answering workflows |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and setup instructions |

## Setup

### 1. Create a Python environment

From the repository folder on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements specify minimum versions rather than a tested, pinned dependency set. Compatibility should be checked in the environment used to run the project.

### 2. Configure Neo4j and Gemini

Use a running Neo4j instance and an existing database accessible to your account. Create a local `.env` file:

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=replace_with_your_password
NEO4J_DATABASE=neo4j
GEMINI_API_KEY=replace_with_your_api_key
```

For a remote Neo4j instance, use its supplied connection URI and database name. The code defaults to `graphragdb` when `NEO4J_DATABASE` is omitted; it does not create that database. The account needs permission to create the chunk constraint and write document records.

Keep `.env`, virtual environments, and generated bytecode out of Git. Suggested `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

### 3. Correct the upload-statistics variable

In the supplied `rag_orchestrator.py`, the assignment to `stats` is commented out, but the success response still references it. In `process_and_store_pdf()`, enable this line after checking that chunks were extracted:

```python
stats = self.pdf_processor.get_chunk_statistics(chunks)
```

Without this change, the method can store chunks and then fail while building its success response.

### 4. Launch the application

```powershell
python app.py
```

Open `http://localhost:7860` in your browser. The supplied launch configuration binds to `0.0.0.0` and does not create a Gradio sharing link. For access restricted to the local machine, change `server_name` to `127.0.0.1`.

The application initializes Neo4j and its AI components at startup. Ensure the required credentials and services are available before launching.

## Using the Interface

1. Open **Upload Documents**, select a PDF with extractable text, and choose **Process PDF**.
2. Open **Query Knowledge Base**, enter a question, and select the number of passages to retrieve.
3. Review the answer and supporting passages.
4. Use **Manage Database** to inspect indexed documents or delete their chunks by exact source filename.

Example questions include “What are the main topics discussed?” and “What conclusions are drawn?” Their answers depend on the documents uploaded; they are not measured project results.

## Configuration in the Supplied Code

| Setting | Value |
|---|---|
| Embedding model identifier | `models/text-embedding-004` |
| Generation model identifier | `gemini-2.0-flash` |
| Minimum paragraph length | More than 20 characters |
| Default retrieval count | 5 |
| Maximum API attempts | 3 |
| Vector index dimensions | 768 |
| Web interface port | 7860 |

The model identifiers reflect the supplied files; their current API availability has not been verified. If changing the embedding model, verify its output dimensions and regenerate stored embeddings and the index as needed. The quota values displayed in the app's System Info tab are hardcoded text, not account limits queried from the provider.

## Implementation Scope and Limitations

**Neo4j is used as an embedding store in this version.** The source creates `Chunk` nodes but does not extract entities, create relationships, traverse a knowledge graph, or use LangChain or LangGraph. A repository named `graphrag-neo4j-assistant` can host this implementation, but the implemented retrieval workflow is vector RAG in Neo4j.

The code attempts to create a cosine vector index, but `search_similar()` does not query that index. It scans stored chunks and computes a dot-product score. Dot product is equivalent to cosine similarity only for normalized vectors; the code does not explicitly normalize them.

Additional limits:

- OCR is disabled, so scanned PDFs may not produce useful text.
- Paragraph chunks have no overlap or maximum-length control.
- Questions are independent; chat history is not maintained.
- Source metadata does not preserve PDF page numbers.
- Upload handling assumes the Gradio filepath value exposes `.name`; verify this against the installed Gradio version and use the path directly if it is returned as a plain string.
- Document IDs use the filename stem and chunk number, so different files sharing a stem can overwrite records. Re-uploading a shorter document does not remove leftover old chunks.
- No relevance threshold, automated evaluation, authentication, or user-specific document isolation is implemented.
- Document text and questions are sent to Gemini for embedding or generation.

## Possible Extensions

Add entity and relationship extraction for graph-based retrieval, use the Neo4j vector index for search, improve chunking and document identifiers, preserve page references, evaluate retrieval and answer quality, and add conversation history or access controls where needed.

## Author

**Ritika Dharamkar**

- [GitHub](https://github.com/RitikaDharamkarJ)
- [LinkedIn](https://www.linkedin.com/in/ritikadharamkar/)
