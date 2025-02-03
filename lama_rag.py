
import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS  
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import *


load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY','xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

groq_model = ChatGroq(model_name="llama-3.2-1b-preview",groq_api_key=groq_api_key)

generic_prompt = """
                    Answer the following questions based on given context only.
                    Please provide most accurate response based on question.
                    <context>
                    {context}
                    </context>
                """

prompt = ChatPromptTemplate.from_messages([("system",generic_prompt),("user","{input}")])


def create_vector_embeddings_ollama():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research") #data ingestion step
        st.session_state.docs  = st.session_state.loader.load() #documents loading
        st.session_state.text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        
user_prompt = st.text_input("Enter the query from the documents")
if st.button("Document Embedding"):
    create_vector_embeddings_ollama()
    st.write("Vector db is ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(groq_model,prompt)
    retriver  =  st.session_state.vectors.as_retriever()
    retriver_chain = create_retrieval_chain(retriver,document_chain)
    response = retriver_chain.invoke({'input':user_prompt})
    st.write(response['answer'])
    


