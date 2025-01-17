import streamlit
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

## Langsmith Tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")



llm = Ollama(model="gemma2:2b")

streamlit.title("What's Running in yout mind")

text_query = streamlit.text_input("Query Please")

if text_query:
    chatprompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful AI Bot"),
        ("human",'Question {question}'),
    ])
    question =chatprompt.format_prompt(question=text_query).to_messages()
    response = llm.invoke(question)
    streamlit.write(response)