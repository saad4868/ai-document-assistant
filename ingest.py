import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

def load_doc(doc_path=r"D:\Data Structure\RAG_FOR_BEGIN\docs"):
    print("-"*60)
    print("Load the Documents")
    print("-"*60)
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"NOT FILES IN FOLDER")
    loader=DirectoryLoader(
        path=doc_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    pdfload=DirectoryLoader(
        path=doc_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()+pdfload.load()
    
    if len(documents) == 0:
        raise FileNotFoundError("No text files found")
    print(f"len of the dcumnet{len(documents)}")
    return documents
def split_doc(documents, chunksize=500, chunkoverlape=10):
    print("-" * 60)
    print("SPLIT THE DOCUMENT")
    print("-" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunksize,
        chunk_overlap=chunkoverlape
    )

    chunks = splitter.split_documents(documents)

    for i, chu in enumerate(chunks):
        print(f"Chunk {i+1}")
        print(f"Data Preview: {chu.page_content}")
        print(f"Metadata: {chu.metadata}")
        print()

    print(f"Total Chunks Created: {len(chunks)}")

    return chunks
def embeddi_store(chunks):
    print("-" * 60)
    print("Creating embeddings and store in  database")
    print("-" * 60)
    presistent_directory="db\chroma_db"
    embedding_model=HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2")
    db=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
         persist_directory=presistent_directory
    )
    print("Vector store created successfully!")
    print(f"Saved at: {presistent_directory}")

    return db

def main():
 
    document = load_doc()
    chunks = split_doc(document)
    embed=embeddi_store(chunks)
if __name__ == "__main__":
    main()

