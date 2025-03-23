from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logfire
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
logfire.configure(token=os.getenv("Logfire_Write_Token"))
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
    deps_type=KnowledgeDeps,
    system_prompt="""
      You are a resume details fetcher chatbot.
      Follow these guidelines:
        - ALWAYS search the knowledge base that is provided in the context between <context> </context> tags to answer user questions.
        - Provide accurate candidate information based ONLY on the information retrieved from the knowledge base. 
        - Never make assumptions or provide information not present in the knowledge base.
        - If information is not found in the knowledge base, politely acknowledge this.
        -Fetch candidate name based on the name provided in <context> </context>
    """
)

def mergeHits(hits: list):
  delimiter = "\n" # Define a delimiter
  return delimiter.join(["data: " + hit.page_content+" candidate: " + os.path.splitext(os.path.basename(hit.metadata.get('source')))[0] for hit in hits])

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
  print("systemm prompt being updated")
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
  # queries = ["nlp guys", 
  #            "his address", 
  #            "John Doe's projects list from Spring 2023", 
  #            "john's tech skills list", 
  #            "Ian Hannson"]

  while(True):
  #for user_message in queries:
      user_message = input()
      if(user_message == 'q'):
        break
      with logfire.span("Calling model") as span:
        knowledgeDeps = KnowledgeDeps(vector_db= vector_db, query=user_message)
        response = agent.run_sync(user_message, message_history=messages, deps= knowledgeDeps)
        console.print(response.data, style="cyan", end="\n\n")
        messages+=response.new_messages()
    
if __name__ == '__main__':
  main()