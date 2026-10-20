import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# 1. Streamlit page setup
st.set_page_config(page_title="Chat with PDF",layout="wide")

st.title("Chat with PDF")
st.write("Upload a PDF and ask questions about its content.")


# 2. Define the structured answer
class PDFAnswer(BaseModel):
    answer: str = Field(description="Answer based on the PDF")
    found_in_document: bool = Field(description="Whether the answer is supported by the PDF")
    source_pages: list[int] = Field(description="Relevant PDF page numbers, starting from 1")


parser = PydanticOutputParser( pydantic_object=PDFAnswer)


# 3. Initialize the embedding model and LLM
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    llm = ChatOllama(model="llama3.2:1b",temperature=0 )
    return embeddings, llm


# 4. Create the prompt
prompt = PromptTemplate(
    template="""
You answer questions using the provided PDF context.

Rules:
- Answer only using information supported by the context.
- If the answer is not present, set found_in_document to false.
- Return a valid JSON object, not a JSON schema.
- source_pages must contain the actual page numbers shown in the context.
- If the answer is absent, use an empty source_pages list.

{format_instructions}

Context:{context}

Question:{question}
""",
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# 5. Upload PDF
uploaded_file = st.file_uploader("Upload your PDF",type=["pdf"])


# 6. Process the PDF
if uploaded_file is not None:
    if (
        st.session_state.get("uploaded_file_name")
        != uploaded_file.name
    ):
        st.session_state.pop("retriever", None)
        st.session_state.pop("uploaded_file_name", None)

    if st.button("Process PDF"):
        os.makedirs("uploads", exist_ok=True)

        pdf_path = os.path.join(
            "uploads",uploaded_file.name)

        with open(pdf_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        try:
            with st.spinner("Reading and processing PDF..."):
                documents = PyPDFLoader(pdf_path).load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
                split_documents = splitter.split_documents(documents)

                if not split_documents:
                    st.error("No readable text was found in this PDF.")
                    st.stop()

                embeddings, llm = load_models()

                # Fresh in-memory database for this uploaded PDF
                vector_store = Chroma.from_documents(documents=split_documents,embedding=embeddings,collection_name="pdf_" + str(abs(hash(uploaded_file.name))))

                st.session_state.retriever = (vector_store.as_retriever(search_kwargs={"k": 4}))
                st.session_state.llm = llm
                st.session_state.uploaded_file_name = (uploaded_file.name)
                st.success(f"PDF processed! Created {len(split_documents)} text chunks.")

        except Exception as e:
            st.error(f"Could not process PDF: {e}")


# 7. Ask questions
if "retriever" in st.session_state:
    st.subheader("Ask a question")

    question = st.chat_input("For example: What programming languages are mentioned?")

    if question:
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding the answer..."):
                    retriever = st.session_state.retriever
                    llm = st.session_state.llm

                    docs = retriever.invoke(question)

                    context = "\n\n".join(
                        f"[Page {doc.metadata.get('page', 0) + 1}]\n"
                        f"{doc.page_content}"
                        for doc in docs
                    )

                    formatted_prompt = prompt.invoke({
                        "context": context,
                        "question": question
                    })

                    response = llm.invoke(formatted_prompt)
                    result = parser.invoke(response.content)

                st.write(result.answer)

                if result.found_in_document:
                    st.caption(
                        "Found in the document: Yes"
                    )
                else:
                    st.caption(
                        "Found in the document: No"
                    )

                if result.source_pages:
                    st.caption(
                        "Source pages: "
                        + ", ".join(
                            str(page)
                            for page in result.source_pages
                        )
                    )

            except Exception as e:
                st.error(f"Could not generate the answer: {e}")
