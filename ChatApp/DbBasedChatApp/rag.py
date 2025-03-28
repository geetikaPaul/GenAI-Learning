from pydantic_ai.models.gemini import GeminiModel
import os
from dotenv import load_dotenv
from pydantic_ai import RunContext, Agent, Tool
import logfire
from rich.console import Console
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
import sqlite3
from dataclasses import dataclass

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))
class SemanticSearch:
  def __init__(
        self,
        src_dir: str,
        kb_dir: str,
        vector_db_dir: str,
        embedding_model_name: str,
        reranking_model_name: str,
        retriever_top_k: int,
        reranker_top_k: int,
    ):
        self.kb_dir = os.path.join(src_dir, kb_dir)
        self.vector_db_dir = os.path.join(src_dir, vector_db_dir)
        self.retriever_top_k = retriever_top_k
        self.reranker_top_k = reranker_top_k
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
        )
        self.rerank_model = HuggingFaceCrossEncoder(model_name=reranking_model_name)
        self.data_ingestion()
        
  def metadata_func(self, record: dict, metadata: dict) -> dict:
    metadata["source"] = record.get("table")
    metadata["columns"] = record.get("columns")
    return metadata

  def data_ingestion(self):
    print("data ingestion begins....")
    documents = []
    for file in Path(self.kb_dir).glob("*.json"):
              loader = JSONLoader(
                  file_path=file,
                  jq_schema=".[]",
                  content_key=".chunk",
                  is_content_key_jq_parsable=True,
                  metadata_func=self.metadata_func,
              )
              documents.extend(loader.load())
          #print(documents[0])

    vector_db = FAISS.from_documents(
              documents=documents,
              embedding=self.embeddings_model,
              distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
          )

    vector_db.save_local(folder_path=self.vector_db_dir)
    self.vector_db = vector_db
    print("Ingestion of data ends........")
    
  def retrieveContent(self, query: str):
          base_retriever = self.vector_db.as_retriever(
              search_kwargs={"k": self.retriever_top_k}
          )
          compressor = CrossEncoderReranker(
              model=self.rerank_model, top_n=self.reranker_top_k
          )
          compression_retriever = ContextualCompressionRetriever(
              base_compressor=compressor, base_retriever=base_retriever
          )
          results = compression_retriever.invoke(query)
          outputs = []
          for result in results:
              output_data = {
              'table': result.metadata.get('source'),
              'columns': result.metadata.get('columns')
              }
              
              outputs.append(output_data)
          return outputs

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
        5,
        2,
    )
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
        prompt = f"<schema> {ss.retrieveContent(user_message)} </schema> question: {user_message}"
        response = queryGenAgent.run_sync(user_prompt=prompt)
        console.print(response.data, style="cyan", end="\n\n")
        promptWIthQuery = f"query: {response.data} question: {user_message}"
        response = queryRunAgent.run_sync(user_prompt=promptWIthQuery, deps=systemDeps)
        console.print(response.data, style="cyan", end="\n\n")
    
if __name__ == '__main__':
  main()