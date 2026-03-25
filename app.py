import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Step 1: Page Configuration
st.set_page_config(page_title="Vishal's AI Assistant")
st.title("Chat with Vishal's Resume")
st.markdown("Ask me anything about Vishal's experience, skills, or projects!")

# Step 2: Initialize RAG Chain (Cached)
@st.cache_resource
def init_rag_chain():
    load_dotenv()

    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore=PineconeVectorStore(
        index_name="pdf-bot",
        embedding=embeddings
    )

    retriever=vectorstore.as_retriever(search_kwargs={"k":3})

    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        verbose=True
    )

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

    verbose=True
    question_answer_chain=create_stuff_documents_chain(llm,prompt)
    verbose=True
    return create_retrieval_chain(retriever,question_answer_chain)

rag_chain=init_rag_chain()

# Step 3: Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages=[]

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Step 4: Chat input and Logic
if prompt:=st.chat_input("Ask about Vishal's Projects..."):
    # User message
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI message
    with st.chat_message("assistant"):
        with st.spinner("Searching resume..."):
            response=rag_chain.invoke({"input":prompt})
            answer=response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role":"assistant","content":answer})