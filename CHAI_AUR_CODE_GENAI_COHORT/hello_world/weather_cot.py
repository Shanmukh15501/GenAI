# pip install openai python-dotenv
# no tools -- pure prompting with chain of thought
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Weather via pure prompting (no tools) ──────────────────────────────────────
# The model reasons step by step through the question using only its training.
# No tool calls, no live data -- just prompting.

SYSTEM = """You are a helpful weather assistant that reasons step by step.

For each response output ONLY a JSON object with exactly two keys:
{
  "step":    "<one of: analyse | think | limitations | result>",
  "content": "<your reasoning for this step>"
}

Progress through the steps in order. Stop at result."""

city = input("Enter city: ")
print()

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": f"What is the weather in {city}?"},
]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
    )

    reply = response.choices[0].message.content
    data  = json.loads(reply)

    print(f"  {data['step']:12}: {data['content']}")

    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user",      "content": "next step"})

    if data["step"] == "result":
        print("\ndone.")
        break
