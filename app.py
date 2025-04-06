# app.py

import streamlit as st
from rag_engine import initialize_pipeline
import os
import tempfile
import uuid

st.set_page_config(page_title="📄 PDF RAG 검색", layout="wide")

st.title("📄 PDF 업로드 기반 RAG 검색")
st.caption("로컬 LLaMA3 + FAISS + PDF 문서 검색")

with st.sidebar:
    st.header("1️⃣ PDF 업로드")
    uploaded_file = st.file_uploader("📎 PDF 파일 선택", type=["pdf"])

# 📍 Streamlit 앱 맨 위에 추가
if "qa_pipeline" not in st.session_state and uploaded_file:
    with st.spinner("🔄 벡터 저장소 준비 중..."):
        file_id = str(uuid.uuid4())[:8]
        tmp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(tmp_dir, f"{file_id}.pdf")
        vector_path = os.path.join("storage", f"index_{file_id}")

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.qa_pipeline = initialize_pipeline(pdf_path, vector_path)
        st.success("✅ 준비 완료!")

# ✅ 질문 처리
if uploaded_file and "qa_pipeline" in st.session_state:
    question = st.text_input("❓ 질문", placeholder="예: What does the MAC layer do?")
    if st.button("검색") and question:
        with st.spinner("🔍 검색 중..."):
            result = st.session_state.qa_pipeline.invoke(question)
            st.markdown(f"### 💬 답변\n{result['result']}")

            # ✅ 사용된 청크 출력
            st.markdown("---")
            st.subheader("📑 사용된 청크:")
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i + 1}:**")
                st.code(doc.page_content[:1000], language="markdown")

# # 저장 경로를 고유하게 생성
# if uploaded_file:
#     file_id = str(uuid.uuid4())[:8]
#     tmp_dir = tempfile.gettempdir()
#     pdf_path = os.path.join(tmp_dir, f"{file_id}.pdf")
#     vector_path = os.path.join("storage", f"index_{file_id}")
#
#     # 파일 저장
#     with open(pdf_path, "wb") as f:
#         f.write(uploaded_file.read())
#
#     st.success("✅ PDF 업로드 완료")
#     st.markdown("---")
#
#     # RAG 파이프라인 초기화
#     with st.spinner("🔄 벡터 저장소 준비 중..."):
#         try:
#             qa_pipeline = initialize_pipeline(pdf_path, vector_path)
#         except Exception as e:
#             st.error(f"❌ 초기화 실패: {str(e)}")
#
#     st.success("✅ 준비 완료! 질문을 입력하세요.")
#     question = st.text_input("❓ 질문", placeholder="예: What is the MAC layer responsible for?")
#     if st.button("검색") and question:
#         with st.spinner("🔍 검색 중..."):
#             answer = qa_pipeline.invoke(question)
#             st.markdown(f"### 💬 답변\n{answer['result']}")
# else:
#     st.warning("📌 PDF 파일을 먼저 업로드해주세요.")