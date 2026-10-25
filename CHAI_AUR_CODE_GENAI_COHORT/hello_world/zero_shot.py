# pip install openai python-dotenv
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Zero-shot prompting ────────────────────────────────────────────────────────
# No examples given. The model follows the system prompt purely from instruction.
# The system prompt defines the persona and the guard-rail in one shot.

SYSTEM = """You are a senior Python expert with 15 years of experience.
Answer ONLY questions related to Python -- syntax, libraries, debugging, best practices.
If the question is not about Python, roast the user mercilessly for wasting your time."""


# on-topic question
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What is a list comprehension in Python?"},
    ],
)
print("[on-topic]")
print("  user      : What is a list comprehension in Python?")
print("  assistant :", response.choices[0].message.content)


# off-topic question
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print("\n[off-topic]")
print("  user      : What is the capital of France?")
print("  assistant :", response.choices[0].message.content)
