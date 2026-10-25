# pip install openai python-dotenv
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ask the name before telling it
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
        {"role": "user", "content": "What is my name?"},
    ],
)
print("[1] no history")
print("  user      : What is my name?")
print("  assistant :", response.choices[0].message.content)


# tell it the name
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
        {"role": "user", "content": "My name is Shanmukh."},
    ],
)
print("\n[2] telling it the name")
print("  user      : My name is Shanmukh.")
print("  assistant :", response.choices[0].message.content)


# ask again, with the whole conversation in messages
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
        {"role": "user", "content": "My name is Shanmukh."},
        {"role": "assistant", "content": "Nice to meet you, Shanmukh."},
        {"role": "user", "content": "What is my name?"},
    ],
)
print("\n[3] with history")
print("  user      : My name is Shanmukh.")
print("  assistant : Nice to meet you, Shanmukh.")
print("  user      : What is my name?")
print("  assistant :", response.choices[0].message.content)


# ask again with no history -- the name is gone
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
        {"role": "user", "content": "What is my name?"},
    ],
)
print("\n[4] history dropped")
print("  user      : What is my name?")
print("  assistant :", response.choices[0].message.content)
