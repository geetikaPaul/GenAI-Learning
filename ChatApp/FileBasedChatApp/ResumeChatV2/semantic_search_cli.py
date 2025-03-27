from dotenv import load_dotenv
import os
from semantic_searcher_with_rerank import SemanticSearcherWithRerank
from rich.console import Console


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
        console.print(result, style="cyan", end="\n\n")


if __name__ == "__main__":
    main()