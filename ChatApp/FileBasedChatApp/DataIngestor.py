import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

load_dotenv(override=True)

class DataIngestor:
  documents: list
  text_splitter: RecursiveCharacterTextSplitter | CharacterTextSplitter
  
  def getTextSplitter(type: str, chunk_size: int):
    if(type == 'recursive'):
      return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function= len)
    else:
      return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, length_function= len)

  def loadPdf(kb_dir: str):
    for file in Path(kb_dir).glob("*.pdf"):
        print(file.name)
        loader = PyPDFLoader(file)
        documents.extend(loader.load_and_split(text_splitter))
    #print(len(documents))
    #print(documents)
    
  def loadCsv(kb_dir: str):
      loader = CSVLoader(Path(kb))
      documents.extend(loader.load_and_split(text_splitter))
      #print(len(documents))
      #print(documents)
      return documents
    
  embedding_models = HuggingFaceEmbeddings(
    model_name = os.getenv("HF_EMBEDDINGS_MODEL"),
    encode_kwargs = {"normalize_embeddings": True},
    model_kwargs = {"token": os.getenv("HuggingFace_AccessToken")}
  )

  def ingest(destination_dir: str):
    vector_db = FAISS.from_documents(documents=documents, embedding=embedding_models,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
    # print(vector_db.index) # used for faster search
    #print(vector_db.docstore) # actual documents = original text + metadata
    # print(vector_db.index_to_docstore_id) # index - document store mappings

    vector_db_dir = os.path.expanduser(destination_dir)
    vector_db.save_local(folder_path=vector_db_dir)
    
  def main():
    src_dir = "~/genAI/ChatApp/FileBasedChatApp/data"
    dest_dir = "~/genAI/ChatApp/FileBasedChatApp/data/index/faissResume"
    file_type = input('ingest pdf or csv files: ')
    documents = []
    if(file_type == 'pdf'):
      documents = loadPdf(src_dir)
    else:
      documents = loadCsv(src_dir)
    ingest(documents, dest_dir)
    
if __name__ == "__main__":
  DataIngestor.main()