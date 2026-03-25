# import os
# import pypdf
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# def ingest_docs():
#     # Step 0: Configuration and API Setup
#     load_dotenv()
#     google_api_key=os.getenv("GOOGLE_API_KEY")
#     pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     DATA_PATH="./data/resume.pdf"

#     # Step 1: PDF Loading
#     print(f"--- Loading {DATA_PATH} ---")
#     reader=pypdf.PdfReader(DATA_PATH)
#     text=""
#     for page in reader.pages:
#         text+=page.extract_text() + "\n"
    
#     #Wrap text into a Langchain Document Object
#     raw_docs=[Document(page_content=text, metadata={"source": DATA_PATH})]

#     # Step 3: Context_Aware Chunking
#     print("--- Splitting text into overlapping chunks ---")
#     text_splitter=RecursiveCharacterTextSplitter(
#         chunk_size=600,
#         chunk_overlap=100,
#         length_function=len
#     )

#     docs=text_splitter.split_documents(raw_docs)

#     # Step 4: Embedding Configuration
#     print("--- Connecting to Google Embedding Model ---")
#     embeddings=GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=google_api_key
#         )
    
#     # Step 5: Vectorization and Storage
#     index_name="pdf-bot"

#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=768,  # for google-gemini-embedding-001
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws",region="us-east-1")
#         )
    
#     vector_store=PineconeVectorStore.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         index_name=index_name
#     )





# if __name__=="__main__":
#     ingest_docs()



import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

def ingest_docs():

    load_dotenv()

    # Step 1: Load and Split
    loader=PyPDFLoader("data/resume.pdf")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=70,
    )
    chunks=text_splitter.split_documents(data)

    # Step 2: Setup Embeddings
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Step 3: Pinecone Cloud Setup
    pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name="pdf-bot"

    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Step 4: Upload chunks to cloud
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print("--- PDF uploaded to Pinecone Cloud ---")


if __name__=="__main__":
    ingest_docs()