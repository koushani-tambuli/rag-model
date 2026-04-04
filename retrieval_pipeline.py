import os
from langchain_chroma import Chroma
from google import genai
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory="db/chroma_db"

# Get API Key from .env
api_key = os.getenv("GENIE_API_KEY")

if not api_key:
    raise ValueError("GENIE_API_KEY not found! Check your .env file.")

client = genai.Client(api_key=api_key)

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

#Load embeddings and vector store
embedding_model = GeminiEmbeddings(model="gemini-embedding-001") 

db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# Search for relevant document
query="When I was born?"

#retriever= db.as_retriever(search_kwargs={"k":4})

retriever=db.as_retriever(
     search_type="similarity_score_threshold",
     search_kwargs={
         "k" : 5,
         "score_threshold" : 0.3 #only return chunks with cosine similarity 0.3
     }
 )

relevant_docs= retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("---- Context ----")
for i,doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n{doc.page_content}\n") 