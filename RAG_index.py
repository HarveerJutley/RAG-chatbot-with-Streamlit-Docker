#Main Pipeline

#imports
import os
import sys
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def load_documents(folder_path):
    docs = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8-sig")
        elif filename.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            continue 

        docs.extend(loader.load())

    return docs


def run_rag_pipeline(folder_path):
#Step 1:Load documents
    print(f"\n[1/4] Loading documents: {folder_path}")
    docs = load_documents(folder_path)
    print(f"loaded {len(docs)}document(s)")
#Step 2:Chunk documents
    print(f"\n[2/4] Chunking documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 250,
        chunk_overlap = 30,
        length_function = len,
        separators= ["\n\n","\n"," ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")
#Step 3:Initialise embeddings
    print(f"\n[3/4] Initialise embedding model")
    embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        openai_api_key = os.getenv("OPENAI_API_KEY")
    )
    print(f" Embedding model ready")
#Step 4:Create vector store 
    print (f"\n[4/4] Create vector store")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f" Vector store created with {len(chunks)} chunks")

    return vectorstore

#Main Execution
if __name__ == "__main__":
    folder_path = "C:/Users/bob/HarveerWork/RAG_Project/docs"
    vectorstore = run_rag_pipeline(folder_path)

    print("\n🔍 Ask a question (type 'exit' to quit):")

    while True:
        query = input("\nYour question: ")

        if query.lower() == "exit":
            break

        results = vectorstore.similarity_search(query,k=5)

        seen = set()

        for i, result in enumerate(results, 1):
            content = result.page_content

    # Skip duplicates
            if content in seen:
                continue

            seen.add(content)

            print(f"\nResult {i}:")
            print("\n" + "="*50)
            print(content)
            print(f"Source: {result.metadata}")
            print("="*50)

            