
import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime 
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
import random
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from io import StringIO
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
import shutil
import chromadb


#If there is no chat_history, then the input is just passed directly to the retriever. If there is chat_history, then the prompt and LLM will be used to generate a search query. That search query is then passed to the retriever.
#create_history_aware_retriever
#A history-aware retriever in LangChain is essential to provide contextually relevant responses by considering previous interactions. It ensures continuity in multi-turn conversations, allowing the model to retrieve more accurate information based on both the current and past queries. This enhances personalization and efficiency in ongoing dialogues.

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
collection_name = "your_collection_name"



load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY','xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

groq_model = ChatGroq(model_name="llama-3.1-8b-instant",groq_api_key=groq_api_key)

import random
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader  # Correct import from langchain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

delete_session = st.button(label="Delete Session", type="secondary", icon=None, disabled=False, use_container_width=False)


if delete_session:
    persistent_client = chromadb.PersistentClient()
    persistent_client = None
    
    del persistent_client
    
    st.session_state.clear()
    # Check if the directory exists
    persist_directory = r'D:\PROJECT_PRACTISE_DIRS\GenAI\Chroma'

    if os.path.exists(persist_directory):
        # Loop through all files and subdirectories in the directory
        for filename in os.listdir(persist_directory):
            file_path = os.path.join(persist_directory, filename)
            # If it's a file, remove it
            if os.path.isfile(file_path):
                os.remove(file_path)
            # If it's a directory, remove the entire directory (including subdirectories)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        st.write("All files and subdirectories removed.")
    else:
        st.write("The specified directory does not exist.")
    st.write("Session cleared")

    
def read_documents():
    uploaded_files=[]
    uploaded_files = st.file_uploader(label="Please upload PDF files", type=['pdf'], accept_multiple_files=True)
    if uploaded_files and 'read' not in st.session_state:
        documents = []    
        # Loop through uploaded files and extract pages
        persist_directory = r'D:\PROJECT_PRACTISE_DIRS\GenAI\Chroma'
        temporary_file = persist_directory + "\\" + "tempo" + str(random.randint(1,1000)) + ".pdf"
        file_path = temporary_file
        for uploaded_file in uploaded_files:
            # Load the PDF directly from memory
            bytes_data = uploaded_file.getvalue()
            
            with open(temporary_file,'wb') as file:
                file.write(bytes_data)
                loader = PyPDFLoader(file_path)
            
            documents.extend(loader.load())

        # Prepare embeddings and text splitting
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Split documents into chunks
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
        
        # Display for debugging purposes (optional)
        
        # Create vector store and retriever
        persist_directory='D:\\PROJECT_PRACTISE_DIRS\\GenAI\\Chroma'
        collection_name = "Chroma"

        st.session_state.vectors = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings,collection_name=collection_name,persist_directory=persist_directory)
        st.session_state.retreiver = st.session_state.vectors.as_retriever()
        st.session_state.read = True
        
        return st.session_state.read  # Indicating that documents have been read

    elif uploaded_files and st.session_state.read:
        st.write("Already Embedded no need to read")
        return st.session_state.read
    else:
        return False 

def get_session_history(session_id: str):
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
            
    return st.session_state.store[session_id]
        

result = read_documents()
if result:
    import random
    session_history=get_session_history('abc7')
    
    user_input = st.text_input("Your question:")
    session_history.add_user_message(user_input)
    if user_input:
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )


        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        #Create a chain that takes conversation history and returns documents.
        #https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html


        history_aware_retriever = create_history_aware_retriever(groq_model, st.session_state.retreiver, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )   

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        #Create a chain for passing a list of Documents to a model.

        question_answer_chain = create_stuff_documents_chain(groq_model, prompt)

        #Create retrieval chain that retrieves documents and then passes them on.


        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":"abc7"}
            },  # constructs a key "abc123" in `store`.
        )

        st.write("Assistant:", response['answer'])
