import operator
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city using wttr.in."""
    response = requests.get(f"https://wttr.in/{city}?format=3")
    return response.text


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


tools = [get_weather, add_numbers]

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, operator.add]


def chatbot(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


def chat():
    print("Chatbot ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        final = None
        for step in graph.stream({"messages": [HumanMessage(content=user_input)]}):
            for node, state in step.items():
                if node == "tools":
                    for msg in state["messages"]:
                        print(f"[Tool: {msg.name}] → {msg.content}")
                elif node == "chatbot":
                    final = state["messages"][-1]
        if final:
            print(f"AI: {final.content}\n")

SAVE_GRAPH_DIAGRAM = False

if SAVE_GRAPH_DIAGRAM:
    from pathlib import Path
    _graph_preview = graph_builder.compile()
    diagram_path = Path(__file__).parent / "tools.png"
    diagram_path.write_bytes(_graph_preview.get_graph().draw_mermaid_png())
    print(f"Graph diagram saved to {diagram_path}")

if __name__ == "__main__":
    chat()
