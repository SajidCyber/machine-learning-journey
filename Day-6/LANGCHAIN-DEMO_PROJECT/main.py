import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # 1. Define the PDF path (Ensure this file exists at this path)
    pdf_path = 'D:\\Agen-Day1\\Day-6\\LANGCHAIN-DEMO_PROJECT\\Agentic_AI_10.pdf' 
    
    if not os.path.exists(pdf_path):
        print(f"Error: The file {pdf_path} was not found. Please update the path.")
        return

    print("Loading PDF...")
    # 2. Load PDF pages using LangChain's PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Successfully loaded {len(pages)} pages.")

    print("\nSplitting text into chunks...")
    # 3. Split the documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(pages)
    print(f"Total Chunks Created: {len(chunks)}")

    print("\nGenerating embeddings and creating FAISS Vector Database...")
    # 4. Load the HuggingFace embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # 5. Build and populate the FAISS Vector Database with chunks
    db = FAISS.from_documents(chunks, embedding_model)
    print("FAISS Vector Database Created Successfully!")

    print("\n--- Running Similarity Search Test ---")
    # 6. Execute a sample similarity query
    query = "What is agentic ai?"
    print(f"Query: '{query}'\n")
    
    results = db.similarity_search(query)
    
    # 7. Print the top matching result text
    if results:
        print("Top Matching Result Content:")
        print(results[0].page_content)
    else:
        print("No matching results found.")

if __name__ == "__main__":
    main()
