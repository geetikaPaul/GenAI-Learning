from pydantic_ai.models.gemini import GeminiModel
import os
from dotenv import load_dotenv
from pydantic_ai import RunContext, Agent, Tool
from utility import Expense
import logfire
from rich.console import Console

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

agent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              ),
            system_prompt= """
                  Return SQL query for given question using <schema>
                  Table: Expense
                  Id (INTEGER)
                  StoreName (TEXT)
                  Category (TEXT)
                  Amount (DECIMAL)
                </schema>
          """
      )

def main():
  console = Console()
  console.print(
        "Welcome to resume Bot. How may I assist you today?",
        style="cyan",
        end="\n\n",
    )
  
  while(True):
      user_message = input()
      if(user_message == 'q'):
        break
      with logfire.span("Calling model") as span:
        response = agent.run_sync(user_prompt=user_message)
        console.print(response.data, style="cyan", end="\n\n")
    
if __name__ == '__main__':
  main()