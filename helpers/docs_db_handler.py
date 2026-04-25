import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from helpers.indexer import split_docs

def load_docs(data_folder):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    doc_loader = PyPDFDirectoryLoader(data_folder)
    return doc_loader.load()

def init_db(chunks, embeddings_model, folder_path, embeddings):
    faiss_path = os.path.join(folder_path, "index.faiss")
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings_model)
        os.makedirs(folder_path, exist_ok=True)
        vectorstore.save_local(folder_path)
    return vectorstore

def add_db_docs(vectorstore, data_path, db_path, embeddings_model):
    documents = load_docs(data_path)
    for document in documents:
        content = document.page_content
        embedding = embeddings_model.embed_query(content)
        result = vectorstore.similarity_search_by_vector(embedding, k=3)
        if not result:
            chunks = split_docs([document])
            vectorstore.add_documents(chunks)
    vectorstore.save_local(db_path)
