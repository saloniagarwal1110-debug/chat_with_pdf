import os
import streamlit as st

from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Chat with My PDF",
    page_icon="📄",
    layout="centered"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #111827;
    color: white;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #A78BFA;
}

.section {
    font-size: 18px;
    font-weight: bold;
    color: white;
    margin-top: 25px;
}

[data-testid="stChatInput"] {
    background-color: #1E293B;
}

[data-testid="stChatInput"] textarea {
    color: white !important;
    -webkit-text-fill-color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. TITLE AND UPLOAD FOLDER
# ==========================================

st.markdown(
    '<div class="title">📄 Chat with My PDF</div>',
    unsafe_allow_html=True
)

UPLOAD_FOLDER = "uploaded_files"
DB_FOLDER = "my_chroma_db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# 4. PYDANTIC OUTPUT MODEL
# ==========================================

class PDFAnswer(BaseModel):
    answer: str = Field(
        description="Answer to the user's question based on the PDF"
    )


# ==========================================
# 5. INITIALIZE LLM
# ==========================================

@st.cache_resource
def load_llm():
    model = ChatOllama(
        model="llama3.2:1b",
        temperature=0
    )

    return model.with_structured_output(
        PDFAnswer,
        method="json_schema"
    )


try:
    structured_llm = load_llm()
except Exception as e:
    st.error(f"Could not initialize Ollama: {e}")
    st.stop()


# ==========================================
# 6. PROMPT AND CHAIN
# ==========================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful PDF question-answering assistant.

Rules:
1. Answer using only the provided PDF context.
2. Give a clear and simple answer.
3. If the answer is not present in the context,
   say: "The answer is not available in the provided PDF."
4. Return the answer in the required structured format.
5. Do not return a JSON schema as the answer.

PDF Context:
{context}
"""
    ),
    ("human", "{question}")
])

chain = prompt | structured_llm


# ==========================================
# 7. SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = None


# ==========================================
# 8. PDF UPLOAD
# ==========================================

st.markdown(
    '<div class="section">📁 Upload your PDF</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


# ==========================================
# 9. PROCESS PDF
# ==========================================

if uploaded_file:

    pdf_path = os.path.join(
        UPLOAD_FOLDER,
        os.path.basename(uploaded_file.name)
    )

    # Use the file content as well as the filename
    # to identify a changed upload.
    import hashlib

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.processed_pdf != file_hash:

        try:
            with st.spinner("Processing your PDF..."):

                # Save uploaded PDF
                with open(pdf_path, "wb") as f:
                    f.write(file_bytes)

                # Load PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                if not documents:
                    st.error("No pages found in this PDF.")
                    st.stop()

                # Split text
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                chunks = splitter.split_documents(documents)

                if not chunks:
                    st.error(
                        "No readable text found. "
                        "This PDF may contain scanned pages."
                    )
                    st.stop()

                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )

                # Create a fresh collection for this uploaded PDF
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="pdf_" + file_hash[:20],
                    persist_directory=DB_FOLDER
                )

                # MMR retriever
                st.session_state.retriever = (
                    vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 3,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )
                )

                st.session_state.processed_pdf = file_hash
                st.session_state.messages = []

            st.success("PDF uploaded and processed successfully!")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.stop()


# ==========================================
# 10. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==========================================
# 11. ASK QUESTIONS
# ==========================================

st.markdown(
    '<div class="section">💬 Ask your question</div>',
    unsafe_allow_html=True
)

query = st.chat_input("Ask anything about your PDF...")


# ==========================================
# 12. GENERATE ANSWER
# ==========================================

if query:

    if st.session_state.retriever is None:

        st.warning("Please upload a PDF first.")

    else:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        try:
            with st.chat_message("assistant"):

                with st.spinner("Finding your answer..."):

                    # Retrieve relevant chunks
                    relevant_docs = (
                        st.session_state.retriever.invoke(query)
                    )

                    # Build context
                    context = "\n\n".join(
                        doc.page_content
                        for doc in relevant_docs
                    )

                    if not context.strip():
                        answer = (
                            "The answer is not available "
                            "in the provided PDF."
                        )
                    else:
                        # Generate structured answer
                        result = chain.invoke({
                            "context": context,
                            "question": query
                        })

                        # Pydantic object or compatible dict
                        if isinstance(result, PDFAnswer):
                            answer = result.answer
                        elif isinstance(result, dict):
                            answer = result.get(
                                "answer",
                                "Unable to generate an answer."
                            )
                        else:
                            answer = str(result)

                    # Display answer
                    st.write(answer)

                    # Save answer to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

        except Exception as e:
            st.error(
                f"Error generating answer: {e}"
            )


# ==========================================
# 13. FOOTER
# ==========================================

st.markdown(
    """
    <p style='text-align:center; color:#9CA3AF; margin-top:40px;'>
        Powered by Generative AI | Built with Streamlit
    </p>
    """,
    unsafe_allow_html=True
)