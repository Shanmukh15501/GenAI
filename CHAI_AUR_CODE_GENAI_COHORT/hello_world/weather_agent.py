# pip install openai python-dotenv requests
import os
import json

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Tool definition ────────────────────────────────────────────────────────────
# Tell the model what tools it can call and what args they take.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current temperature and weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name e.g. London, Hyderabad",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


# ── Tool execution ─────────────────────────────────────────────────────────────
# wttr.in is a free weather API -- no key needed.

def get_weather(city):
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url, timeout=5)
    data = response.json()
    current = data["current_condition"][0]
    return {
        "city":        city,
        "temp_c":      current["temp_C"],
        "temp_f":      current["temp_F"],
        "feels_like_c": current["FeelsLikeC"],
        "description": current["weatherDesc"][0]["value"],
        "humidity":    current["humidity"],
    }


# ── Agent loop ─────────────────────────────────────────────────────────────────
# Same while/break pattern as chain_of_thought.py.
# Loop runs until the model stops calling tools and gives a final text reply.

city = input("Enter city: ")

messages = [
    {"role": "system", "content": "You are a helpful weather assistant. Use the get_weather tool to fetch live data before answering."},
    {"role": "user",   "content": f"What is the weather in {city}?"},
]

print()

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    message = response.choices[0].message

    # model wants to call a tool
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        args      = json.loads(tool_call.function.arguments)

        print(f"  [agent calling tool] get_weather(city={args['city']})")

        weather = get_weather(args["city"])

        # append the tool call and its result into messages
        messages.append(message)
        messages.append({
            "role":         "tool",
            "tool_call_id": tool_call.id,
            "content":      json.dumps(weather),
        })

    # model has the data and is giving the final answer -- break
    else:
        print(f"  assistant : {message.content}")
        print("\ndone.")
        break
