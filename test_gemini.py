import os

from dotenv import load_dotenv
from google import genai


# Load the API key from the .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")


# Connect to Gemini
client = genai.Client(api_key=api_key)


# Send a simple test request
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Reply with exactly: Gemini connection works!"
)


# Print Gemini's response
print(response.text)