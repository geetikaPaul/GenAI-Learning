from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from rich.console import Console
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from langchain_community.vectorstores import FAISS
from pydantic_ai import RunContext
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

load_dotenv(override=True)

class KnowledgeDeps(BaseModel):
  vector_db: FAISS
  query: str
  
  class Config:
        arbitrary_types_allowed = True

client = genai.Client(api_key=os.getenv("Gemini_API_Key"))
agent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
    ),
    system_prompt="You are a resume details fetcher chatbot.",
)

def mergeHits(hits: list):
  delimiter = "\n" # Define a delimiter
  return delimiter.join([hit.page_content for hit in hits])

embedding_models = HuggingFaceEmbeddings(
  model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
  encode_kwargs = {"normalize_embeddings": True},
  model_kwargs = {"token": os.getenv("HuggingFace_AccessToken")}
)

rerank_model = HuggingFaceCrossEncoder(model_name=os.getenv("HF_ReRanker_MODEL"))

vector_db_dir = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/index/faissResume")
vector_db = FAISS.load_local(
    folder_path=vector_db_dir,
    embeddings=embedding_models,
    allow_dangerous_deserialization=True,
)

@agent.system_prompt
def add_rag_prompt(ctx: RunContext[KnowledgeDeps]) -> str:
  base_retriever = ctx.deps.vector_db.as_retriever(search_kwargs = {"k":5})
  compressor = CrossEncoderReranker(model=rerank_model, top_n=2)
  compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
  )
    
  hits = compression_retriever.invoke(ctx.deps.query)
  prompt = mergeHits(hits)
  
  return f"<context> {prompt} </context>"
  
def main():
  console = Console()
  console.print(
        "Welcome to resume Bot. How may I assist you today?",
        style="cyan",
        end="\n\n",
    )

  messages = []

  while(True):
      user_message = input()
      if(user_message == 'q'):
        break
      knowledgeDeps = KnowledgeDeps(vector_db= vector_db, query=user_message)
      response = agent.run_sync(user_message, message_history=messages, deps= knowledgeDeps)
      console.print(response.data, style="cyan", end="\n\n")
      messages+=response.new_messages()
    
if __name__ == '__main__':
  main()