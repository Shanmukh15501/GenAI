import os
import valkey

VALKEY_HOST = os.getenv("VALKEY_HOST", "localhost")
VALKEY_PORT = int(os.getenv("VALKEY_PORT", 6379))

client = valkey.Valkey(host=VALKEY_HOST, port=VALKEY_PORT, decode_responses=True)
