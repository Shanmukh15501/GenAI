# pip install openai python-dotenv
# needs OPENAI_API_KEY in a .env file next to this script or at the project root
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_here = Path(__file__).resolve().parent
for _candidate in (_here / ".env", _here.parent / ".env"):
    if _candidate.is_file():
        load_dotenv(_candidate)
        break

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = "hi hello i am shanmkukh"

res = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
)

vector = res.data[0].embedding

print("text      :", text)
print("model     :", res.model)
print("length    :", len(vector))        # 1536 for -small, 3072 for -large
print("first 10  :", vector[:10])
print("tokens    :", res.usage.total_tokens)
