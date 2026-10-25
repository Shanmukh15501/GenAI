# pip install openai python-dotenv
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Few-shot prompting ─────────────────────────────────────────────────────────
# Give the model 2-3 example Q&A pairs before the real question.
# The model learns the expected tone, format, and depth from the examples --
# no separate instruction needed.

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a strict Python expert. Answer ONLY Python questions. If the question is not directly about Python, refuse and say nothing else."},

        # example 1
        {"role": "user",      "content": "What is a tuple?"},
        {"role": "assistant", "content": "A tuple is an immutable ordered sequence. Example: coords = (10, 20). Use it when the data should not change."},

        # example 2
        {"role": "user",      "content": "What is a dictionary?"},
        {"role": "assistant", "content": "A dict maps keys to values. Example: user = {'name': 'Shanmukh', 'age': 22}. Lookups are O(1)."},

        # example 3
        {"role": "user",      "content": "What is a lambda?"},
        {"role": "assistant", "content": "A lambda is a one-line anonymous function. Example: square = lambda x: x**2. Use it for short throwaway functions."},

        # real question -- model copies the pattern from the examples above
        {"role": "user",      "content": "Explain Static variables in python"},
    ],
)
print("[few-shot] real question : Explain Static variables in python?")
print("assistant :", response.choices[0].message.content)


# off-topic -- should get a hard refusal
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a strict Python expert. Answer ONLY Python questions. If the question is not directly about Python, refuse and say nothing else."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
)
print("\n[off-topic] real question : What is the capital of France?")
print("assistant :", response.choices[0].message.content)
