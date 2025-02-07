import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
import streamlit as st
import time
import numpy as np
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from guardrails import Guard
from guardrails.hub import DetectPII
from guardrails.hub import CompetitorCheck, ToxicLanguage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()

delete_session = st.button(label="Delete Session", type="secondary", icon=None, disabled=False, use_container_width=False)


groq_api_key=os.getenv('GROQ_API_KEY')

llm = ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

# Setup system message
system_prompt = """You are a highly skilled software developer with expertise in coding, debugging, and creating test cases. Your primary focus is to assist with coding-related inquiries and problem-solving for software development, specifically in Python and Java.

### Instructions:
- **Primary Role:** Only answer questions related to software development, coding, debugging, and programming concepts. Do not answer questions unrelated to these topics.
- **Avoid:** Any answers related to general knowledge, personal queries, entertainment, or anything not directly related to programming.
- **Response Focus:** Your answers should be focused solely on coding-related topics, such as debugging, code optimization, explaining programming concepts, and software development best practices. 
"""



def get_session_history(session_id: str):
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
            
    return st.session_state.store[session_id]

with_message_history = RunnableWithMessageHistory(llm, get_session_history)

config = {"configurable": {"session_id": "abc2"}}


response = with_message_history.invoke(
    [SystemMessage(content=system_prompt)],
    config=config,
)



user_input = st.text_input("Enter Your Queries related to Coding", '')

if input:
    response = with_message_history.invoke(
    [HumanMessage(content=user_input)],
    config=config,
)
    
    st.write(response.content)

    st.divider()  # 👈 Draws a horizontal rule
    st.header("Using Guard Rails")            
    guard = Guard().use_many(DetectPII(pii_entities=["EMAIL_ADDRESS","PHONE_NUMBER"],
                                        on_fail="fix"
                                        ),
                            CompetitorCheck(["Apple", "Microsoft", "Google"], on_fail="fix"),
                            ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="fix")
                            )

    
    response = response.content
    

    prompt = ChatPromptTemplate.from_template(f'Just Print the Input Provided {response}')

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser | guard.to_runnable()
    
    res = chain.invoke({"response":response})

    st.write(res)
