## load the saved pdf 
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

pdf_path=r"C:\Users\Win 11\Desktop\Gen ai\chat_with_pdf project\uploads\Saloni_Agarwal_Resume.pdf"

loader=PyPDFLoader(pdf_path)
documents=loader.load()

print("total pages",len(documents))
print(documents[0].page_content)    


## split the pdf into small chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=20)
chunks=text_splitter.split_documents(documents)

print("total chunks:",len(chunks))
print(chunks[0].page_content)


