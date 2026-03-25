import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Step 1: Setup
load_dotenv()
embeddings=GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Step 2: Connect to ChromaDB
db=Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Step 3: Ask a test question
user_query="What are the technical skills Vishal has mentioned in his resume?"

# Step 4: Search the database
print(f"--- Searching for: {user_query} ---")
docs=db.similarity_search(user_query, k=2)  #Get the top 2 matching chunks

# Step 5: Display the result
for i,doc in enumerate(docs):
    print(f"\n--- Found Chunk : {i+1} ---")
    print(doc.page_content[:300]+"...")     #Shows the first 300 characters of the match

print("\n Retrieval Successful!")