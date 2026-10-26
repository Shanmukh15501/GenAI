from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
import os
import json

from openai import OpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from mem0 import Memory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str


mem0_config = {
    "version": "v1.1",
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": OPENAI_API_KEY,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "langgraph_memories",
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "neo4j_password",
        },
    },
}

memory = Memory.from_config(mem0_config)


def chat_node(state: State):
    user_id = state.get("user_id", "default_user")
    user_message = state["messages"][-1]["content"] if isinstance(state["messages"][-1], dict) else state["messages"][-1].content

    relevant_memories = memory.search(query=user_message, filters={"user_id": user_id}, limit=5)
    memories = [
        f"ID: {m.get('id')} Memory: {m.get('memory')}"
        for m in relevant_memories.get("results", [])
    ]

    system_prompt = f"""You are a memory-aware assistant which responds to the user with context.
You are given past memories and facts about the user.

Memory of the user:
{json.dumps(memories)}"""

    result = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    assistant_reply = result.choices[0].message.content

    memory.add(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_reply},
        ],
        user_id=user_id,
    )

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=assistant_reply)]}


graph_builder = StateGraph(State)
graph_builder.add_node("chat_node", chat_node)
graph_builder.add_edge(START, "chat_node")
graph_builder.add_edge("chat_node", END)

graph = graph_builder.compile()


def main():
    user_id = input("Enter your user ID (or press Enter for 'default_user'): ").strip()
    if not user_id:
        user_id = "default_user"

    print(f"\nChat started for user: {user_id}")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("> ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": query}],
                "user_id": user_id,
            }
        )

        print("=" * 50)
        print(result["messages"][-1].content)
        print("=" * 50)


SAVE_GRAPH_DIAGRAM = False

if SAVE_GRAPH_DIAGRAM:
    from pathlib import Path
    diagram_path = Path(__file__).parent / "MEMORY_graph_diagram.png"
    diagram_path.write_bytes(graph.get_graph().draw_mermaid_png())
    print(f"Graph diagram saved to {diagram_path}")

if __name__ == "__main__":
    main()
