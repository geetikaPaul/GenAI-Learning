from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import logfire
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.AgentBuilder import get_prompt

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

file_path = os.path.expanduser("~/genAI/Translator/data/voiceClip1.mp3")

agent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
    ),
    system_prompt="You are a language translator.",
)

with logfire.span("Calling Gemini model") as span:
    response = agent.run_sync(get_prompt("translate to English", file_path))
    print(response.data)
    