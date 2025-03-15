import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)

embedding_models = HuggingFaceEmbeddings(
  model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
  encode_kwargs = {"normalize_embeddings": True},
  model_kwargs = {"token": os.getenv("HuggingFace_AccessToken")}
)

vector_db_dir = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/index/faissResume")
vector_db = FAISS.load_local(
    folder_path=vector_db_dir,
    embeddings=embedding_models,
    allow_dangerous_deserialization=True,
)

queries = [
    "Name",
    "ABC Company",
    "NLP Algorithm",
    "Soft: Teamwork, Creative Problem Solving, Comm"
]

for query in queries:
    hits = vector_db.similarity_search_with_score(query, k=1)

    print("\nQuery:", query)
    print("Top most similar chunks in corpus/knowledge base:")
    # print(hits)
    for hit in hits:
        print(hit[0].page_content, "(Score: {:.4f})".format(hit[1]))
        print()