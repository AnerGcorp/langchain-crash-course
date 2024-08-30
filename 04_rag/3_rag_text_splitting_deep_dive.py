import os
from dotenv import load_dotenv

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load the environment variables
load_dotenv()

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found, please check the path.")

# Read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Function to create and persist vector store
def create_vector_store(docs, store_name):
    persist_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persist_dir):
        print(f"Creating vector store: {store_name}")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_dir
        )
        print(f"Vector store created and persisted at {persist_dir}")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize")

# 1. Character-based Splitting
# Splits text into chunks based on a specified number of characters.
# Useful for consistent chunk sizes regardless of content structure.
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

# 2. Sentence-based Splitting
# Splits text into chunks based on sentences, ensuring chunks end at sentence boundaries.
# Ideal for maintaining semantic coherence within chunks.
print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")

# 3. Token-based Splitting
# Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# Useful for transformer models with strict token limits.
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

# 4. Recursive Character-based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")

# 5. Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("\n--- Using Custom Splitting ---")

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # Custom logic for splitting text
        return text.split("\n\n")  # Example: split by paragraphs


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, "chroma_db_custom")

def query_vector_store(query, store_name):
    persist_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Vector store {store_name} not found. Please check the path is correct.")
    
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.1}
    )
    relevant_docs = retriever.invoke(query)
    
    print(f"\n--- Relevant Documents from {store_name} ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}: \n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")

# Define the user's query
query = "What is the tragedy of Romeo and Juliet?"

# Query the vector stores
query_vector_store(query, "chroma_db_char")
query_vector_store(query, "chroma_db_sent")
query_vector_store(query, "chroma_db_token")
query_vector_store(query, "chroma_db_rec_char")
query_vector_store(query, "chroma_db_custom")