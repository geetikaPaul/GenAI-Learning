from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
from rich.console import Console
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.mistral import MistralModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.AgentBuilder import get_prompt
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

ImgDescAgent = Agent(
    model=MistralModel(
                model_name="pixtral-12b-2409", api_key=os.environ["MISTRAL_API_KEY"]
                ),
    system_prompt="Give description of the immage.",
)

chattingAgent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
    ),
    system_prompt="You are a chatbot.",
)

def main():
  with logfire.span("Calling model") as span:
    console = Console()
    console.print(
          "Welcome to GP's Chat Bot. How may I assist you today?",
          style="cyan",
          end="\n\n",
      )
  
    file_path = os.path.expanduser("~/genAI/ChatApp/sevenWonders.png") 
    response = ImgDescAgent.run_sync(get_prompt(user_prompt="What's in the image?", file_path=file_path))
    
    while(True):
      user_message = input()
      if(user_message == 'q'):
        break
      response = chattingAgent.run_sync(user_message, message_history=response.all_messages())
      console.print(response.data, style="cyan", end="\n\n")
    
if __name__ == '__main__':
  main()