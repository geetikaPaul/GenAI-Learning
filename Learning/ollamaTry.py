# install ollama by downloading the right file from https://github.com/ollama/ollama
# ollama run llama3.2
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))
logfire.instrument_openai()

agent = Agent(
    model=OpenAIModel(
        model_name="llama3.2", base_url="http://localhost:11434/v1"
    ),
    system_prompt="You are a helpful assistant.",
)

response = agent.run_sync("Write a haiku about recursion in programming.")
print(response.data)

response = agent.run_sync("What is recursion in programming.")
print(response.data)

print(response.usage())