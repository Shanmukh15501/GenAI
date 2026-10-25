# pip install tiktoken
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "hi hello i am shanmkukh"

# encode: text -> token ids
ids = enc.encode(text)
print("text   :", text)
print("ids    :", ids)
print("tokens :", [enc.decode([i]) for i in ids])
print("count  :", len(ids))

# decode: token ids -> text
print("decoded:", enc.decode(ids))
