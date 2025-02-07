import os
import streamlit as st
import validators
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY=""

llm = ChatGroq(model="deepseek-r1-distill-llama-70b",groq_api_key=GROQ_API_KEY)

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="gpt-4o")



chunks_prompt="""
        Please summarize the below Info:
        Info:`{text}'
        Summary:
    """
map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompt)

final_prompt=''' Provide the final summary of the entire data with these important points.
                Add a proper heading and and add points.
                Info:{text}
        '''
final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)



def summarize_document_technique(llm,chain_type,verbose,docs,map_prompt=None,combine_prompt=None):
    
    if chain_type == 'stuff':
        chain=load_summarize_chain(llm,chain_type,verbose)
        result = chain.run(docs)
    elif chain_type == 'map_reduce':
        chain=load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
            )
        result = chain.run(docs)
    else:
        chain = load_summarize_chain(
                llm=llm,
                chain_type="refine",
                verbose=True,                
                )
        result = chain.run(docs)
    return result
        

st.header("Data Summarisation Techniques")
url = st.text_input("Provide a valid Web url / Youtube url to Extract the information")
if validators.url(url):
    url=url.strip()
    technique = st.radio("Choose One of the following Technique",
                         ["Stuff Document Chain", "Map Reduce Chain", "Refine Chain"]
                        )
    if 'youtube' in url:
        loader=YoutubeLoader.from_youtube_url(url,add_video_info=False)
        docs = loader.load()
        final_documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)

    else:
        loader = WebBaseLoader(url)
        docs = loader.load()
        final_documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)

        
    if technique == "Stuff Document Chain":
        chain_type = 'stuff'
        verbose = True
        result = summarize_document_technique(llm,chain_type,verbose,docs)
        st.write(result)
    elif technique == "Map Reduce Chain":                
        chain_type="map_reduce"
        
        verbose=True
        
        result = summarize_document_technique(llm,chain_type,verbose,final_documents,map_prompt_template,final_prompt_template)
        
        st.write(result)
    else:
        chain_type="refine"
        verbose=True
        result = summarize_document_technique(llm,chain_type,verbose,final_documents,map_prompt_template,final_prompt_template)
        st.write(result)
