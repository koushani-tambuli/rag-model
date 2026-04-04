import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader # Read files
from langchain_text_splitters import CharacterTextSplitter  # For chunking
from google import genai #  Embedding model
from langchain_chroma import Chroma # vector database
from dotenv import load_dotenv # load env variable
from langchain.embeddings.base import Embeddings

load_dotenv()

# Get API Key from .env
api_key = os.getenv("GENIE_API_KEY")

if not api_key:
    raise ValueError("GENIE_API_KEY not found! Check your .env file.")

# Initialize the NEW Google GenAI client
client = genai.Client(api_key=api_key)

# Test the key immediately - if this fails, the key is still being blocked
# List models
#models = client.models.list()
#print("Available models and their supported methods:\n")
#for m in models:
 #   print(m.name, m.supported_actions)

# 3. Custom Gemini Embeddings class using the new SDK

class GeminiEmbeddings(Embeddings):
    def __init__(self, model="gemini-embedding-001"):  # Use a valid model
        self.model = model

    def embed_documents(self, texts):
        # Generate embeddings for a list of texts
        result = client.models.embed_content(
            model=self.model,
            contents=texts
        )
        # Extract embedding vectors
        return [emb.values for emb in result.embeddings]

    def embed_query(self, text):
        # Generate embedding for a single query
        result = client.models.embed_content(
            model=self.model,
            contents=text
        )
        return result.embeddings[0].values
    

def load_documents(docs_path="docs"):
    "Load all text files from doc directory"
    print(f"Loading documents from {docs_path}...")

    #Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} doesnot exist. Please create it and add your company files.")
    
    #Load all .txt files from the docs directory
    loader = DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}: ")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}....")
        print(f" metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=100):
    print("\nSplitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        print(f"Total chunks created: {len(chunks)}")

        # Preview first 5 chunks
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)}")
            print(f"Content: ")
            print(chunk.page_content)
            print("--" * 50)
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    "Create and persist ChromaDB vector store"
    print("Creating and embeddings and storing in ChromaDB")

    embedding_model = GeminiEmbeddings(model="gemini-embedding-001")

    #Create ChromaDB vector store
    print("----Creating Vector Store----")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("----Finished creating vector store----")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    try:
        print("Main Function")
        # LOAD FILES
        documents= load_documents(docs_path="docs")
        #Chunking
        chunks= split_documents(documents)
        #Embeddinh and Storing
        vectorstore=create_vector_store(chunks)
        print("Ingestion Pipeline Complete!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()