from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from rich.console import Console
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

load_dotenv(override=True)

print(os.getenv("Gemini_API_Key"))
client = genai.Client(api_key=os.getenv("Gemini_API_Key"))

agent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
    ),
    system_prompt="You are a chatbot.",
)
result = agent.run_sync('How are you?')  
print(result.data)