from pydantic_ai.models.gemini import GeminiModel
import os
from dotenv import load_dotenv
from pydantic_ai import RunContext, Agent
#import logfire
from rich.console import Console
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import sqlite3
from dataclasses import dataclass
from semantic_search_with_rerank import SemanticSearch
from Queue import FixedSizeList

load_dotenv(override=True)
#logfire.configure(token=os.getenv("Logfire_Write_Token"))

@dataclass
class SystemDeps:
    db_path: str
    
    
queryGenAgent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              ),
            system_prompt= """
                  Generate SQL query for given question using <schema> </schema>
                """
      )

queryRunAgent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              ),
            system_prompt= """
                  Run the given query to generate response to the question
                """,
          deps_type= SystemDeps
      )

@queryRunAgent.tool
def get_expenses(ctx: RunContext[SystemDeps], query: str):
    """ Run query and generate the results"""
    with sqlite3.connect(ctx.deps.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows

def main():
  systemDeps = SystemDeps(db_path=os.path.expanduser("~/genAI/Chatapp/DbBasedChatApp/data/Chinook_Sqlite.sqlite"))
  src_dir = os.path.expanduser("~/genAI/Chatapp/DbBasedChatApp/data")
  kb_dir = "metadata"
  vector_db_dir = "index"
  ss = SemanticSearch(
        src_dir,
        kb_dir,
        vector_db_dir,
        os.getenv("HF_EMBEDDINGS_MODEL"),
        os.getenv("HF_ReRanker_MODEL"),
        20,
        5,
    )
  console = Console()
  console.print(
        """
        Welcome to text-sql-response Bot. I have data for albums, track, genre, invoice, employee, customer
        How may I assist you today?
        """,
        style="cyan",
        end="\n\n",
    )
  
  messagesSql = FixedSizeList(10)
  messagesData = FixedSizeList(10)
  
  while(True):
      user_message = input()
      if(user_message == 'q'):
        break
      #with logfire.span("Calling model") as span:
      prompt = f"<schema> {ss.retrieveContent(user_message)} </schema> question: {user_message}"
      response = queryGenAgent.run_sync(user_prompt=prompt) #, message_history=messagesData.get_all())
      console.print(response.data, style="cyan", end="\n\n")
      #messagesSql.add(response.all_messages())
        
      promptWIthQuery = f"query: {response.data} question: {user_message}"
      response = queryRunAgent.run_sync(user_prompt=promptWIthQuery, deps=systemDeps) #, message_history=messagesData.get_all())
      console.print(f"output: {response.data}", style="cyan", end="\n\n")
      #messagesData.add(response.all_messages())
    
if __name__ == '__main__':
  main()