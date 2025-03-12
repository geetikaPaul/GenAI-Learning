import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import CSVLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

kb = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/username.csv")
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function= len)
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators=';'
)

documents = []
loader = CSVLoader(Path(kb))
documents.extend(loader.load_and_split(text_splitter))
print(len(documents))
print(documents)

embedding_models = HuggingFaceEmbeddings(
  model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
  encode_kwargs = {"normalize_embeddings": True},
  model_kwargs = {"token": os.getenv("HuggingFace_AccessToken")}
)

vector_db = FAISS.from_documents(documents=documents, embedding=embedding_models,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
# print(vector_db.index) # used for faster search
#print(vector_db.docstore) # actual documents = original text + metadata
# print(vector_db.index_to_docstore_id) # index - document store mappings

vector_db_dir = os.path.expanduser("~/genAI/ChatApp/FileBasedChatApp/data/faissCSV")
vector_db.save_local(folder_path=vector_db_dir)