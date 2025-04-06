from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def pdf_to_chunks(pdf_path):
    text = extract_text(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_chunks = splitter.split_text(text)
    return pdf_chunks

import faiss
import numpy as np

def create_faiss_index(chunks, embed_model):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Ollama 모델 세팅
llm = OllamaLLM(model="llama3.1:8b")

# 텍스트 문서 객체화
def make_documents(chunks):
    return [Document(page_content=c) for c in chunks]

# 전체 RAG 파이프라인
def build_qa_pipeline(chunks, embed_model):
    docs = make_documents(chunks)
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    db = FAISS.from_documents(docs, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa

chunks = pdf_to_chunks("sample.pdf")
embed_model = SentenceTransformer("all-mpnet-base-v2")
qa_pipeline = build_qa_pipeline(chunks, embed_model)

response = qa_pipeline.invoke("What's the MAC layer responsibility?")
print(response["result"])