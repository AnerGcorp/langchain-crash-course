import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load the environment variables
load_dotenv()

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_dir = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store..")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found, please check the path.")

    # Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print(f"\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: {docs[0].page_content}\n")

    # Create the OpenAI embeddings
    print("\n--- Creating OpenAI embeddings ---")
    embeddigns = OpenAIEmbeddings(
        model="text-embedding-3-small"
    ) # Update a valid embedding model if needed
    print("Embeddings created successfully")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddigns, persist_directory=persistent_dir)
    print(f"Vector store created and persisted at {persistent_dir}")
else:
    print("Vector store already exists. No need to initalize")