from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.AgentBuilder import get_agent

load_dotenv(override=True)
#logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
#logfire.instrument_openai()

system_prompt = """
    You are helpful assistant.
    Use 'add' tool to add two integers.
    Use 'mul' tool to multiply two integers.
"""
agent = get_agent(system_prompt= system_prompt)

@agent.tool_plain
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b


@agent.tool_plain
def mul(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b


response = agent.run_sync("10 plus 20")
print(response.data)

response = agent.run_sync("10 * 20")
print(response.data)