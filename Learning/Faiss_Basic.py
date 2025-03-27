import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain.schema import Document

load_dotenv(override=True)

embeddings_model = HuggingFaceEmbeddings(
    model_name=os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs={"normalize_embeddings": True},
    model_kwargs={"token": os.getenv("HuggingFace_AccessToken")},
)

# texts

chunks = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

vector_db = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)
# print(vector_db.index)
# print(vector_db.docstore)
# print(vector_db.index_to_docstore_id)

result = vector_db.similarity_search("A man eating pasta", k=2)
print(result)

# documents - article/ long text with metadata

documents = [
    Document(page_content="I love programming in Python.", metadata={"title": "Python Programming", "source": "Article 1"}),
    Document(page_content="FAISS is great for similarity search.", metadata={"title": "Using FAISS", "source": "Article 2"}),
    Document(page_content="Machine learning models are powerful.", metadata={"title": "Machine Learning", "source": "Article 3"})
]

faiss_index = FAISS.from_documents(documents=documents,embedding=embeddings_model)
result = faiss_index.similarity_search("what is faiss used for?", k=1)
print(result)