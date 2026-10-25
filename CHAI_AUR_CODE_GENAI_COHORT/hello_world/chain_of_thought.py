# pip install openai python-dotenv
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Chain of Thought prompting ─────────────────────────────────────────────────
# Model drives the steps itself. Each response is JSON with "step" and "content".
# Loop breaks only when the parsed step == "result".

SYSTEM = """You are a helpful AI assistant that reasons step by step.

For each response output ONLY a JSON object with exactly two keys:
{
  "step":    "<one of: analyse | breakdown | think | validate | result>",
  "content": "<your reasoning for this step>"
}

Progress through the steps in order. Stop atwww result."""

question = "What is 2 + 2?"
print(f"user : {question}\n")

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user",   "content": question},
]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
    )

    reply = response.choices[0].message.content
    data  = json.loads(reply)

    print(f"  {data['step']:10}: {data['content']}")

    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user",      "content": "next step"})

    if data["step"] == "result":
        print("\ndone.")
        break
