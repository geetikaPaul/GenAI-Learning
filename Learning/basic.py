from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyB52oe7AmtddKWeY7pxMYYDLF_4h9wkp8M")
sys_instruct="You are a cat. Your name is Neko."

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=sys_instruct),
    contents="what is your name?",
)

print(response.text)