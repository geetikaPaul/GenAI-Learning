from dotenv import load_dotenv
from pydantic_ai import Agent
import logfire
import os
from pydantic_ai.models.gemini import GeminiModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant.
    Use 'add' tool to add two integers.
    Use 'mul' tool to multiply two integers.
"""
agent = Agent(
    model=GeminiModel(
                model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
          ),
    system_prompt=system_prompt
)

@agent.tool_plain
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b


@agent.tool_plain
def mul(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b

with logfire.span("Calling model") as span:
  response = agent.run_sync("10 plus 20")
  print(response.data)

  response = agent.run_sync("10 * 20")
  print(response.data)