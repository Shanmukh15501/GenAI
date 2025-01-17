from fastapi import FastAPI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

load_dotenv()

groq_api_key=os.getenv('GROQ_API_KEY')

groq_model = ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

parser = StrOutputParser()


generic_prompt = "Translate into following {language}"

prompt = ChatPromptTemplate.from_messages([("system",generic_prompt),("user","{text}")])

prompt.invoke({"language":"Hindi","text":"Hello this is shanmukh"})

chain3 = prompt | groq_model | parser

chain3.invoke({"language":"Hindi","text":"Hello this is shanmukh"})


#App definition

app = FastAPI(
              title="Lang Serve",
              version="1.0",
              description="A simple application"
              )

add_routes(
    app,
    chain3,
    path="/chain3"
)


if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)
