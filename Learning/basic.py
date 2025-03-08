from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = genai.Client(api_key=os.getenv("Gemini_API_Key"))
sys_instruct="You are a cat. Your name is Neko."

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=sys_instruct),
    contents="what is your name?",
)

print(response.text)