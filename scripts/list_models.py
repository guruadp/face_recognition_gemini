import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Print only models that can generate text
for m in client.models.list():
    # Some entries may not have all fields; keep it robust.
    name = getattr(m, "name", None)
    if name:
        print(name)