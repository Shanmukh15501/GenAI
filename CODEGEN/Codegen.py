import os
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import sqlite3
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import initialize_agent, Tool, AgentType ,tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import create_engine
from pathlib import Path
import sqlite3
from guardrails import Guard
from guardrails.hub import DetectPII
from guardrails.hub import CompetitorCheck, ToxicLanguage


load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo")
 

st.title("🦜🔗 CodeGen Demo")

# Function to connect to the database
def db_connect():
    # Get the absolute path of the database file
    dbfilepath = (Path(__file__).parent / "users.db").absolute()
    
    print("dbfilepath",dbfilepath)
    
    # SQLite connection factory
    creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
    
    print("creator",creator)
    
    # Return SQLDatabase connection
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

# Check if user is already registered or not
if 'active' not in st.session_state or st.session_state.active == False:
    # Prompt for registration if user is not active
    st.header("Please Register Yourself")
    
    # Registration form
    with st.form("my_form"):
        first_name = st.text_input("Enter Your First Name", '')
        last_name = st.text_input("Enter Your Last Name")
        email = st.text_input("Enter Your Email")
        phone_number = st.text_input("Enter Your Phone Number")
        password = st.text_input("Enter Your Password", type="password")  # changed to password type for security
        location = st.selectbox('Location', ['Visakhapatnam', 'HYD', 'Chennai', 'Pune', 'Delhi'])
        rating = st.slider('Rate Your Coding Skills', 1, 5)  # Rate from 1 to 5
        submit = st.form_submit_button('Submit')
    
    # If form is submitted
    if submit:
        # Ensure that required fields are filled
        if not first_name or not last_name or not location or not rating:
            st.error("Please fill in all the required fields.")  # Error message if fields are missing
        else:
            # Mark user as active and store session data
            st.session_state.active = True
            st.write(f"Welcome {first_name} {last_name}!")
            
            db=db_connect()
            st.write("connected to database")
            
            
            # Update session state with the provided data
            
            data = {
                'first_name': first_name,
                'last_name': last_name,
                'location': location,
                'email': email,
                'phone_number': phone_number,
                'marks': rating,
                'password': password
            }
            # Construct the SQL insert statement, making sure to format values properly
            columns = ', '.join(data.keys())
            values = ', '.join([f"'{v}'" for v in data.values()])  # Ensure values are wrapped in quotes for strings
            
            ## connect to sqllite
            connection=sqlite3.connect("users.db")

            ##create a cursor object to insert record,create table
            cursor=connection.cursor()

            insert_query = f"INSERT INTO UserManagement ({columns}) VALUES ({values})"
            # Execute the insert query directly
            cursor.execute(insert_query)
            st.session_state.active = True
            st.write("Insertion Complete")
            connection.commit()
            connection.close()
            
            
    else:
        st.session_state.active = False  # User is not registered if form is not submitted yet

else:
    # Placeholder for active user state (optional)
    db=db_connect()
    ser_input = st.text_input('Enter Your Queries Related to User Management Table')
    try:
        agent_executor = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
        response = agent_executor.invoke({'input': user_input})
        st.write("Response: ", response['output'])
        st.divider()  # 👈 Draws a horizontal rule
        st.header("Using Guard Rails")            
        guard = Guard().use_many(
                                DetectPII(
                                            pii_entities=["EMAIL_ADDRESS","PHONE_NUMBER"],
                                            on_fail="fix"
                                            ),
                CompetitorCheck(["Appleu", "Microsoft", "Google"], on_fail="fix"),
                ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail="fix")
                )


        prompt = ChatPromptTemplate.from_template("Just Print the Statement {response}")

        output_parser = StrOutputParser()

        chain = prompt | llm | output_parser | guard.to_runnable()

        res = chain.invoke({"response": response['output']})

        st.write(res)
        st.session_state.active = True
        
    except Exception as e:
        st.write("Error: ", e)  # This will help catch any issues in the request
        


  


