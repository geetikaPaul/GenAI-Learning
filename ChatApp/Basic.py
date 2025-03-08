from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from rich.console import Console
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

load_dotenv(override=True)

client = genai.Client(api_key=os.getenv("Gemini_API_Key"))
agent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
    ),
    system_prompt="You are a chatbot.",
)

def main():
  console = Console()
  console.print(
        "Welcome to GP's Chat Bot. How may I assist you today?",
        style="cyan",
        end="\n\n",
    )
  
  messages = []
  
  while(True):
    user_message = input()
    if(user_message == 'q'):
      break
    response = agent.run_sync(user_message, message_history=messages)
    console.print(response.data, style="cyan", end="\n\n")
    messages+=response.new_messages()
    
if __name__ == '__main__':
  main()