# RAG App with Ollama + LangChain + FAISS

A local Retrieval-Augmented Generation (RAG) application that lets you chat with PDF documents using a fully local LLM stack — no cloud API keys required.

## How It Works

```
PDF files → Chunked → Embedded (MiniLM) → FAISS vector store
User question → Embed → Similarity search → Top-k chunks → LLM (Ollama/Llama3) → Answer
```

Conversation history is preserved per session using LangChain's `RunnableWithMessageHistory`, so the model maintains context across turns.

## Features

- Chat with one or more PDF documents via a Streamlit web UI or CLI
- Fully local — LLM runs via Ollama, embeddings via HuggingFace sentence-transformers
- Duplicate detection — new documents are only indexed if not already in the vector store
- Session memory — conversation history persisted to JSON and reloaded on next run
- File upload — drag-and-drop PDFs directly in the web UI

## Project Structure

```
rag-ollama/
├── app.py                  # CLI entry point
├── webui.py                # Streamlit web UI entry point
├── requirements.txt
├── data/                   # Place PDF files here
├── db/                     # FAISS vector store (auto-created, gitignored)
├── sessions/               # Session JSON history (auto-created, gitignored)
└── helpers/
    ├── __init__.py
    ├── chain_handler.py    # LangChain RAG chain setup
    ├── docs_db_handler.py  # FAISS init, load, dedup logic
    ├── embedder.py         # HuggingFace embeddings wrapper
    ├── indexer.py          # PDF loading + text splitting
    ├── retriever.py        # Vector similarity retrieval
    └── session_handler.py  # Session history load/save
```

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running
- Llama 3 model pulled via Ollama

## Installation

```bash
# 1. Clone
git clone https://github.com/techanvconsulting/rag-ollama.git
cd rag-ollama

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the LLM model via Ollama
ollama pull llama3
```

## Running

### Web UI (recommended)

```bash
streamlit run webui.py
```

Open [http://localhost:8501](http://localhost:8501). Upload PDFs via the sidebar or drop them in `data/` beforehand.

### CLI

```bash
python app.py
```

Type your question at the prompt. Type `exit` to quit.

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| LLM model | `helpers/chain_handler.py` | `llama3` |
| Embedding model | `app.py` / `webui.py` | `sentence-transformers/all-MiniLM-L12-v2` |
| Chunk size | `helpers/indexer.py` | `1000` chars, `80` overlap |
| Retrieved docs (k) | `app.py` / `webui.py` | `5` |
| Ollama base URL | `helpers/chain_handler.py` | `http://127.0.0.1:11434` |

To swap the LLM, change the model name in `helpers/chain_handler.py`:

```python
llm = ChatOllama(model="mistral", base_url="http://127.0.0.1:11434", keep_alive=-1)
```

Any model available via `ollama list` works.

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain-community` | LangChain integrations (Ollama, FAISS, loaders) |
| `langchain-huggingface` | HuggingFace embeddings |
| `faiss-cpu` | Local vector store |
| `sentence-transformers` | Embedding model |
| `streamlit` | Web UI |
| `pypdf` | PDF parsing |
| `langchainhub` | Prompt hub access |

## Reference Documentation

| Resource | Link |
|----------|------|
| Ollama | [ollama.com](https://ollama.com/) |
| Ollama docs | [docs.ollama.com](https://docs.ollama.com) |
| LangChain Python | [docs.langchain.com](https://docs.langchain.com) |
| LangChain FAISS integration | [python.langchain.com/docs/integrations/vectorstores/faiss](https://python.langchain.com/docs/integrations/vectorstores/faiss/) |
| LangChain ChatOllama | [python.langchain.com/docs/integrations/chat/ollama](https://python.langchain.com/docs/integrations/chat/ollama/) |
| RunnableWithMessageHistory | [python.langchain.com/docs/how_to/message_history](https://python.langchain.com/docs/how_to/message_history/) |
| FAISS (Facebook Research) | [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss) |
| all-MiniLM-L12-v2 model card | [huggingface.co/sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) |
| Streamlit docs | [docs.streamlit.io](https://docs.streamlit.io/) |
| pypdf docs | [pypdf.readthedocs.io](https://pypdf.readthedocs.io/) |

## Troubleshooting

**`connection refused` on Ollama** — Start the server first: `ollama serve`

**Empty / "I don't know" answers** — Retrieved chunks may not contain relevant content. Add more PDFs or reduce chunk size in `helpers/indexer.py`.

**`ModuleNotFoundError`** — Always run from the project root (`rag-ollama/`), not from inside `helpers/`.

**Slow first run** — `all-MiniLM-L12-v2` downloads ~120 MB from HuggingFace on first use.

## Roadmap

- [x] Streamlit web UI
- [x] Conversation memory (per-session JSON)
- [x] Duplicate document detection
- [x] Proper Python package structure (`helpers/` as package)
- [ ] Support for `.txt`, `.md`, `.docx` files
- [ ] Model selector in UI
- [ ] Source citation in answers

## Contributing

Contributions welcome. Fork the repo, create a branch, and open a pull request. For larger changes, open an issue first.

## Contact

Open an issue at [github.com/techanvconsulting/rag-ollama](https://github.com/techanvconsulting/rag-ollama/issues).
