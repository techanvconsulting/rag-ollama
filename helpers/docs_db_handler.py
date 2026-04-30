import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from helpers.indexer import split_docs

def load_docs(data_folder):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    doc_loader = PyPDFDirectoryLoader(data_folder)
    return doc_loader.load()

def _indexed_files_path(db_path):
    return os.path.join(db_path, "indexed_files.json")

def _load_indexed_files(db_path):
    path = _indexed_files_path(db_path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return set(json.load(f))
    return set()

def _save_indexed_files(db_path, sources):
    with open(_indexed_files_path(db_path), 'w') as f:
        json.dump(list(sources), f)

def init_db(chunks, embeddings_model, folder_path):
    faiss_path = os.path.join(folder_path, "index.faiss")
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(folder_path, embeddings_model, allow_dangerous_deserialization=True)
    else:
        if not chunks:
            return None
        os.makedirs(folder_path, exist_ok=True)
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
        vectorstore.save_local(folder_path)
        sources = set(c.metadata.get('source', '') for c in chunks)
        _save_indexed_files(folder_path, sources)
    return vectorstore

def add_db_docs(vectorstore, data_path, db_path, embeddings_model):
    if vectorstore is None:
        return
    documents = load_docs(data_path)
    indexed = _load_indexed_files(db_path)
    new_docs = [d for d in documents if d.metadata.get('source', '') not in indexed]
    if not new_docs:
        return
    chunks = split_docs(new_docs)
    vectorstore.add_documents(chunks)
    indexed.update(d.metadata.get('source', '') for d in new_docs)
    _save_indexed_files(db_path, indexed)
    vectorstore.save_local(db_path)
