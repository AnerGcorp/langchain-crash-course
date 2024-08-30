import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load the environment variables
load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_dir}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store..")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Directory {books_dir} not found, please check the path.")
    
    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print(f"\n--- Document Chunk Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create Embeddings
    print("\n--- Creating OpenAI embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embeddings created successfully")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persistent_dir
    )
    print(f"--- Finished creating and persisting vector story ---")
else:
    print("Vector store already exists. No need to initialize.")