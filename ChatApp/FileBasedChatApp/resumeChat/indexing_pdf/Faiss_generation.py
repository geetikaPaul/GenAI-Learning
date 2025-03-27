import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

kb_dir = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/resume")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function= len)

documents = []
for file in Path(kb_dir).glob("*.pdf"):
    print(file.name)
    loader = PyPDFLoader(file)
    documents.extend(loader.load_and_split(text_splitter))
#print(len(documents))
print(documents[0])

embedding_models = HuggingFaceEmbeddings(
  model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
  encode_kwargs = {"normalize_embeddings": True},
  model_kwargs = {"token": os.getenv("HuggingFace_AccessToken")}
)

vector_db = FAISS.from_documents(documents=documents, embedding=embedding_models,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
# print(vector_db.index) # used for faster search
#print(vector_db.docstore) # actual documents = original text + metadata
# print(vector_db.index_to_docstore_id) # index - document store mappings

vector_db_dir = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/index/faissResume")
vector_db.save_local(folder_path=vector_db_dir)