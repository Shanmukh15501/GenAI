from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()
#from langfuse.langchain import CallbackHandler

#langfuse_handler = CallbackHandler()


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = init_chat_model(model_provider="openai", model="gpt-4.1")


def chat_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)

graph_builder.add_node("chat_node", chat_node)

graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)


def compile_graph_with_checkpointer(checkpointer):
    graph_with_checkpointer = graph_builder.compile(checkpointer=checkpointer)
    return graph_with_checkpointer


def main():
    #   mongodb://<username>:<pass>@<host>:<port>
    DB_URI = "mongodb://admin:admin@localhost:27017"
    config = {"configurable": {"thread_id": "2",
                               }}

    with MongoDBSaver.from_conn_string(DB_URI) as mongo_checkpointer:

        graph_with_mongo = compile_graph_with_checkpointer(mongo_checkpointer)

        query = input("> ")

        result = graph_with_mongo.invoke(
            {"messages": [{"role": "user", "content": query}]}, config)

        print("=" * 50)
        print(result["messages"][-1].content)
        print("=" * 50)


SAVE_GRAPH_DIAGRAM = False

if SAVE_GRAPH_DIAGRAM:
    from pathlib import Path
    _graph_preview = graph_builder.compile()
    diagram_path = Path(__file__).parent / "2_langgraph_chat_mongodb_graph_diagram.png"
    diagram_path.write_bytes(_graph_preview.get_graph().draw_mermaid_png())
    print(f"Graph diagram saved to {diagram_path}")

if __name__ == "__main__":
    main()
