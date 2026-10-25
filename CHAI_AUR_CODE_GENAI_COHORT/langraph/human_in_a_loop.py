from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import interrupt, Command
import json

load_dotenv()


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {"query": query})  # This saves the state in DB and kills the graph
    return human_response["data"]


tools = [human_assistance]


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools(tools=tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}


tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")


def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)


def user_chat():
    DB_URI = "mongodb://admin:admin@localhost:27017"
    config = {"configurable": {"thread_id": "220"}}

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
        graph_with_cp = create_chat_graph(mongo_checkpointer)

        while True:
            user_input = input("> ")
            if user_input.strip().lower() in ("exit", "quit", "q"):
                break

            state = State(
                messages=[{"role": "user", "content": user_input}]
            )

            for event in graph_with_cp.stream(state, config, stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()

            snapshot = graph_with_cp.get_state(config)
            print("snapshot.nex",snapshot.next)
            if snapshot.next:
                print("\n[Graph paused — switch to Admin mode to respond]")
                break


def admin_call():
    DB_URI = "mongodb://admin:admin@localhost:27017"
    config = {"configurable": {"thread_id": "220"}}

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:
        graph_with_cp = create_chat_graph(mongo_checkpointer)

        snapshot = graph_with_cp.get_state(config=config)

        if not snapshot.next:
            print("No pending human assistance request.")
            return

        user_query = None
        for task in snapshot.tasks:
            for intr in task.interrupts:
                user_query = intr.value.get("query")

        print("User Has a Query:", user_query)
        solution = input("> ")

        resume_command = Command(resume={"data": solution})

        for event in graph_with_cp.stream(resume_command, config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()


def main():
    while True:
        print("\n[1] User Chat  [2] Admin  [q] Quit")
        choice = input("Choose: ").strip().lower()
        if choice == "1":
            user_chat()
        elif choice == "2":
            admin_call()
        elif choice == "q":
            break
        else:
            print("Invalid choice.")


main()