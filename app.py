import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


pdf_path = (r"C:\Users\Win 11\Desktop\Gen ai\chat_with_pdf project\uploads\Saloni_Agarwal_Resume.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"PDF file not found: {pdf_path}")

## loading pdf
loader = PyPDFLoader(pdf_path)
documents = loader.load()

if not documents:
    raise ValueError("No pages were extracted from the PDF.")

print("PDF loaded successfully!")
print("Number of pages:", len(documents))


## splitting pdf into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
split_documents = text_splitter.split_documents(documents)

if not split_documents:
    raise ValueError("No text chunks were created.")

print("Number of chunks:", len(split_documents))


## embedding 

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Embedding model loaded!")


## CREATE / UPDATE THE CHROMA VECTOR STORE
# A separate database avoids mixing with your old database.



persist_directory = "my_chroma_db_rag"
collection_name = "resume_rag"

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_function,
    persist_directory=persist_directory
)

# Stable IDs update matching chunks on subsequent runs
# instead of repeatedly adding identical chunks.
document_ids = [f"resume-chunk-{i}"
    for i in range(len(split_documents))]

vector_store.add_documents(documents=split_documents,ids=document_ids)

print("Documents stored in Chroma!")

## creating the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
print("Retriever created!")


## define the pydanic output structure
class PDFAnswer(BaseModel):
    answer: str = Field(description="Answer based only on the PDF context" )
    found_in_document: bool = Field(description="Whether the answer is supported by the context")
    source_pages: list[int] = Field(description="Relevant PDF page numbers, starting from 1")


parser = PydanticOutputParser(pydantic_object=PDFAnswer)

## initializing ollama model
llm = ChatOllama(model="llama3.2:1b",temperature=0,format="json")
print("Ollama model configured!")


## creating the prompt template
prompt = PromptTemplate(
    template="""
You answer questions using only the supplied PDF context.

Follow these rules:
1. Answer only with information supported by the context.
2. Do not invent facts.
3. If the answer is supported by the context,
   set found_in_document to true.
4. If the answer is not supported,
   set found_in_document to false.
5. Use an empty source_pages list if the answer is absent.
6. Use the page numbers shown in the context.
7. Return a JSON object containing exactly the required fields.
8. Do not return a JSON schema, Markdown fences, or extra text.

Required JSON format:
{format_instructions}

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()})


## creating the rag question answer function
def answer_pdf_question(question: str) -> PDFAnswer:

    # Retrieve relevant PDF chunks
    docs = retriever.invoke(question)

    if not docs:
        return PDFAnswer(
            answer="No relevant information was found in the PDF.",found_in_document=False,source_pages=[])

    # Add page numbers to the context
    context_parts = []

    for doc in docs:
        page_number = doc.metadata.get("page", 0) + 1

        context_parts.append(f"[Page {page_number}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

    # Prepare the prompt
    formatted_prompt = prompt.format( context=context,question=question)

    # Generate the answer using Ollama
    response = llm.invoke(formatted_prompt)

    # Parse and validate the model's JSON output
    result = parser.parse(response.content)

    # Only retain source pages that actually appear in retrieved context
    available_pages = {doc.metadata.get("page", 0) + 1
        for doc in docs}

    result.source_pages = sorted({
        page for page in result.source_pages
        if page in available_pages})

    return result


## testing the retriever

question = "What certificates are mentioned in my resume?"

relevant_docs = retriever.invoke(question)

print("\n========== RETRIEVED PDF CONTENT ==========")

for doc in relevant_docs:
    print(doc.page_content)
    print("Page:", doc.metadata.get("page", 0) + 1)
    print("-" * 40)


# ============================================================
## Testing the rag chain

print("\n========== QUESTION 1 ==========")

result = answer_pdf_question(
    "What certificates are mentioned in my resume?")

print("Answer:", result.answer)
print("Found in document:", result.found_in_document)
print("Source pages:", result.source_pages)


print("\n========== QUESTION 2 ==========")

result = answer_pdf_question(
    "What programming languages are mentioned in my resume?")

print("Answer:", result.answer)
print("Found in document:", result.found_in_document)
print("Source pages:", result.source_pages)


print("\nRAG pipeline completed!")
