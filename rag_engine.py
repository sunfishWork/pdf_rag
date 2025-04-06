# rag_engine.py

from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import os

def pdf_to_chunks(pdf_path, chunk_size=1000, overlap=200):
    raw_text = extract_text(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(raw_text)

def embed_and_save(chunks, storage_path):
    docs = [Document(page_content=c) for c in chunks]
    embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    db = FAISS.from_documents(docs, embedding=embedding)
    db.save_local(storage_path)

def load_vectorstore(storage_path):
    embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return FAISS.load_local(storage_path, embeddings=embedding, allow_dangerous_deserialization=True)

def build_qa_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = OllamaLLM(model="llama3.1:8b")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, chain_type="map_reduce" )

def initialize_pipeline(pdf_path, storage_path):
    if not os.path.exists(os.path.join(storage_path, "index.faiss")):
        chunks = pdf_to_chunks(pdf_path)
        embed_and_save(chunks, storage_path)
    vectorstore = load_vectorstore(storage_path)
    return build_qa_pipeline(vectorstore)