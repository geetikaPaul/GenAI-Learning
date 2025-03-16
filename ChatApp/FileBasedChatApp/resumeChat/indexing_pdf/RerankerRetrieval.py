import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

load_dotenv(override=True)

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

queries = [
    "Proseware, Inc.",
    "John Doe contact",
    "Ian Hannson skills"
]

for query in queries:
    retriever = vector_db.as_retriever(search_kwargs = {"k":5})
    compressor = CrossEncoderReranker(model=rerank_model, top_n=1)
    compression_retriever = ContextualCompressionRetriever(
          base_compressor=compressor, base_retriever=retriever
    )
    hits = compression_retriever.invoke(query)

    print("\nQuery:", query)
    print("Top most similar chunks in corpus/knowledge base:")
    # print(hits)
    for hit in hits:
        #print(hit[0].page_content, "(Score: {:.4f})".format(hit[1]))
        print(hit.page_content)
        print(hit.metadata.get('source'))
        print()