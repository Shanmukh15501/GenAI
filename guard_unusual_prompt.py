import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from guardrails import Guard
from guardrails.hub import DetectPII
from guardrails.hub import CompetitorCheck, ToxicLanguage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from guardrails.hub import UnusualPrompt
from litellm import completion


load_dotenv()


message  = input("enter your query")

groq_api_key=os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


llm = ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

    
guard = Guard().use_many(
    CompetitorCheck(["Apple", "Microsoft", "Google"], on_fail="fix"),
    UnusualPrompt(llm_callable="groq/llama-3.1-8b-instant",on_fail="exception")
)
prompt = ChatPromptTemplate.from_template(f'{message}')

output_parser = StrOutputParser()

chain = prompt | llm | output_parser | guard.to_runnable()

try:
    res = chain.invoke({'input':message})
    print('res',res)
except Exception as e:
    print("exception as e ",e)





