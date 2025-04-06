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
from pydantic_ai.providers.google_gla import GoogleGLAProvider

load_dotenv(override=True)
#logfire.configure(token=os.getenv("Logfire_Write_Token"))

@dataclass
class SystemDeps:
    db_path: str
    
    
agent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", provider=GoogleGLAProvider(api_key=os.getenv("Gemini_API_Key"))
              ),
            system_prompt= """
                  Given the following question, follow these steps:
                  1. Generate SQL Query: Using the provided <schema></schema> context, identify and create the appropriate SQL query that answers the question.
                  2. Execute SQL Query: Run the generated SQL query using the tool to retrieve the results.
                  3. Convert Results to Natural Language: Once you have the results from the query, convert the information into a brief, easy-to-understand response in natural language, summarizing the answer in a few words.
                  Make sure the final output is a human-readable response based on the query execution, not just the SQL query.
                """,
              deps_type = SystemDeps
      )

@agent.tool
def exec_query(ctx: RunContext[SystemDeps], query: str):
    """ Run query and generate the results"""
    with sqlite3.connect(ctx.deps.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows

def rag():
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
  
  messagesSql = FixedSizeList(10) #rag-semantic search-get relevant old msgs
  messagesData = FixedSizeList(10)
  
  while(True):
      user_message = input()
      if(user_message == 'q'):
        break
      #with logfire.span("Calling model") as span:
      prompt = f"<schema> {ss.retrieveContent(user_message)} </schema> question: {user_message}"
      response = agent.run_sync(user_prompt=prompt, deps=systemDeps) #, message_history=messagesData.get_all())
      console.print(response.data, style="cyan", end="\n\n")
      #messagesSql.add(response.all_messages())
      

def rag_get_response(user_message: str):
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
  prompt = f"<schema> {ss.retrieveContent(user_message)} </schema> question: {user_message}"
  response = agent.run_sync(user_prompt=prompt, deps=systemDeps) #, message_history=messagesData.get_all())
  return response.data
    
if __name__ == '__main__':
  rag()