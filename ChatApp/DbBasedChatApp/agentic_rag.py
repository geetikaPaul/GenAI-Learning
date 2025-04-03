from pydantic_ai.models.gemini import GeminiModel
import os
from dotenv import load_dotenv
from pydantic_ai import RunContext, Agent
import logfire
from rich.console import Console
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import sqlite3
from dataclasses import dataclass
from semantic_search_with_rerank import SemanticSearch
from Queue import FixedSizeList

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

@dataclass
class SystemDeps:
    db_path: str
    ss: SemanticSearch
    
    
agent = Agent(
            model= GeminiModel(
                  model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
              ),
            system_prompt= """
                  Given the following question, follow these steps:
                  0. If the user prompt does not look like a valid question. try to rewrite query
                  1. Get context: Find the required table schema using tool. 
                  2. If no schema found then rewrite query and try to find the schema else move to next step. Try maximum 2 times to get schema
                  3. Augment context: Use the context and user prompt or question to Generate SQL Query: Using the provided <schema></schema> context, identify and create the appropriate SQL query that answers the question.
                  4. Execute SQL Query: Run the generated SQL query using the tool to retrieve the results.
                  5. Convert Results to Natural Language: Once you have the results from the query, convert the information into a brief, easy-to-understand response in natural language, summarizing the answer in a few words.
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

  
@agent.tool
def get_context(ctx: RunContext[SystemDeps], user_prompt: str):
    """Get required table schema for given user_prompt"""
    return f"<schema> {ctx.deps.ss.retrieveContent(user_prompt)} </schemma>"

def main():
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
  systemDeps = SystemDeps(db_path=os.path.expanduser("~/genAI/Chatapp/DbBasedChatApp/data/Chinook_Sqlite.sqlite"), ss= ss)
    
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
      with logfire.span("Calling model") as span:
        prompt = f"<schema> {ss.retrieveContent(user_message)} </schema> question: {user_message}"
        response = agent.run_sync(user_prompt=user_message, deps=systemDeps) #, message_history=messagesData.get_all())
        console.print(response.data, style="cyan", end="\n\n")
        #messagesSql.add(response.all_messages())
    
if __name__ == '__main__':
  main()