from google import genai
import os

api_key = os.getenv("GEMINI_API_KEY")

# Create the Client
client = genai.Client(api_key=api_key)

print("List of available models:")

# Use the client to list models
try:
    for m in client.models.list():
        print(f"- {m.name}")

except Exception as e:
    print(f"Error: {e}")
