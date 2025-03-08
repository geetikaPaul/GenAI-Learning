from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

agent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
    ),
    system_prompt="You are a language translator. Convert given text into Hindi.",
)

with logfire.span("Calling Gemini model") as span:
    response = agent.run_sync("mera naam geetika hai")
    print(response.data)
    