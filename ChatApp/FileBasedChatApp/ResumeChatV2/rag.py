from dotenv import load_dotenv
import os
from semantic_searcher_with_rerank import SemanticSearcherWithRerank
from rich.console import Console
import logfire
from google import genai
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

client = genai.Client(api_key=os.getenv("Gemini_API_Key"))
agent = Agent(
    model=GeminiModel(
                  model_name="gemini-2.0-flash-exp", provider=GoogleGLAProvider(api_key=os.getenv("Gemini_API_Key"))
              ),
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

def get_rag_context(context: str) -> str:
  return f"<context> {context} </context>"

def main():
    load_dotenv(override=True)
    src_dir = os.path.expanduser(
        "~/genAI/ChatApp/FileBasedChatApp/data"
    )
    vector_db_dir = "faissResumeV2"
    kb_dir = "resume"
    embedding_model_name = os.getenv("HF_EMBEDDINGS_MODEL")
    reranking_model_name = os.getenv("HF_ReRanker_MODEL")
    retriever_top_k = 5
    reranker_top_k = 2
    ss = SemanticSearcherWithRerank(
        src_dir,
        kb_dir,
        vector_db_dir,
        embedding_model_name,
        reranking_model_name,
        retriever_top_k,
        reranker_top_k,
    )
    console = Console()
    console.print(
        "Welcome to SemanticSearch On Resumes.  Ask questions and get semantically closest results.",
        style="cyan",
        end="\n\n",
    )
    while True:
        user_question = input(">>")
        if user_question == "q":
            break
        console.print()
        result = ss.retrieveContent(user_question)
        with logfire.span("Calling model") as span:
          prompt = get_rag_context(context= result) + "\n\n query :" + user_question
          response = agent.run_sync(prompt)
        console.print(response.data, style="cyan", end="\n\n")


def rag_get_response(user_question: str):
    load_dotenv(override=True)
    src_dir = os.path.expanduser(
        "~/genAI/ChatApp/FileBasedChatApp/data"
    )
    vector_db_dir = "faissResumeV2"
    kb_dir = "resume"
    embedding_model_name = os.getenv("HF_EMBEDDINGS_MODEL")
    reranking_model_name = os.getenv("HF_ReRanker_MODEL")
    retriever_top_k = 5
    reranker_top_k = 2
    ss = SemanticSearcherWithRerank(
        src_dir,
        kb_dir,
        vector_db_dir,
        embedding_model_name,
        reranking_model_name,
        retriever_top_k,
        reranker_top_k,
    )
    result = ss.retrieveContent(user_question)
    prompt = get_rag_context(context= result) + "\n\n query :" + user_question
    response = agent.run_sync(prompt)
    return response.data

if __name__ == "__main__":
    main()