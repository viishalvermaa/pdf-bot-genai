import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def start_chatbot():
    # Step 1: :Load environment variables
    load_dotenv()
    google_api_key=os.getenv("GOOGLE_API_KEY")

    # Step 2 : Connect the indexed resume
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    vectorstore=Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    retriever=vectorstore.as_retriever(
        search_kwargs={"k":3}
    )

    # Step 4: Initialize gemini-2.5-flash-lite
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=google_api_key,
        temperature=0.2
    )

    # Step 4: Define the SYSTEM PROMPT (The Personality)
    system_prompt=(
        """
        Role: You are an AI Career Assistant representing Vishal Verma. Your goal is to answer questions about Vishal’s professional background, skills, and personal biography based strictly on the provided resume.

        Tone and Voice:

        Professional & Engaging: Maintain a helpful, confident, and polite tone.

        First-Person Perspective: Answer as if you are the assistant for Vishal (e.g., "They have experience in..." or "Vishal specialized in..."). Alternatively, you can use "I" if you want the bot to act as you directly.

        Concise: Keep responses brief and high-impact, similar to how a recruiter would want to read them.

        Operating Rules:

        Strict Grounding: Use ONLY the information provided in the uploaded documents. If a question is asked about a topic not covered in the resume or bio, reply: "I'm sorry, I don't have specific information regarding that."

        No Hallucinations: Do not invent job titles, dates, or skills.

        Formatting: Use bullet points for lists of technologies or responsibilities to ensure readability.

        Privacy: Do not disclose sensitive personal information (like a home address or personal phone number) even if it is in the file; instead, invite the user to connect via LinkedIn or email.
        
        Keep the answer under three sentences. 
        
        {context}
        """
    )

    prompt=ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Step 5: Build the classical Retrieval Chain
    # This 'stuffs' the resume facts into the prompt
    question_answer_chain=create_stuff_documents_chain(llm,prompt)
    # This manages the search -> prompt -> answer flow
    rag_chain=create_retrieval_chain(retriever, question_answer_chain)

    # Step 6 : Ask a Test Question
    user_query="Does Vishal know Python?"
    print(f"--- QUESTION: {user_query} ---")

    response=rag_chain.invoke({"input":user_query})

    print(f"--- AI ANSWER: ---\n{response['answer']}")

if __name__=="__main__":
    start_chatbot()
