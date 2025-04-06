import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticSearcherWithRerank:
    def __init__(
        self,
        src_dir: str,
        kb_dir: str,
        vector_db_dir: str,
        embedding_model_name: str,
        reranking_model_name: str,
        retriever_top_k: int,
        reranker_top_k: int,
    ):
        self.kb_dir = os.path.join(src_dir, kb_dir)
        self.vector_db_dir = os.path.join(src_dir, vector_db_dir)
        self.retriever_top_k = retriever_top_k
        self.reranker_top_k = reranker_top_k
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"token": os.getenv("HUGGING_FACE_TOKEN")},
        )
        self.rerank_model = HuggingFaceCrossEncoder(model_name=reranking_model_name)
        self.ingestJson()

    def ingest(self):
        print("Ingestion of data begins........")
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function= len)

        for file in Path(self.kb_dir).glob("*.pdf"):
            loader = loader = PyPDFLoader(file)
            documents.extend(loader.load_and_split(text_splitter))

        vector_db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings_model,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        vector_db.save_local(folder_path=self.vector_db_dir)
        self.vector_db = vector_db
        print("Ingestion of data ends........")
        
    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["source"] = record.get("source")
        metadata["sectionTitle"] = record.get("sectionTitle")
        return metadata

    def ingestJson(self):
        print("Ingestion of data begins........")
        documents = []
        for file in Path(self.kb_dir).glob("*.json"):
            loader = JSONLoader(
                file_path=file,
                jq_schema=".[]",
                content_key=".chunk",
                is_content_key_jq_parsable=True,
                metadata_func=self.metadata_func,
            )
            documents.extend(loader.load())
        #print(documents[0])

        vector_db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings_model,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

        vector_db.save_local(folder_path=self.vector_db_dir)
        self.vector_db = vector_db
        print("Ingestion of data ends........")

    def retrieve(self, query: str):
        base_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": self.retriever_top_k}
        )
        compressor = CrossEncoderReranker(
            model=self.rerank_model, top_n=self.reranker_top_k
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        results = compression_retriever.invoke(query)
        outputs = []
        for result in results:
            outputs.append(result.metadata["sectionTitle"])
        return outputs

    def retrieveContent(self, query: str):
        base_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": self.retriever_top_k}
        )
        compressor = CrossEncoderReranker(
            model=self.rerank_model, top_n=self.reranker_top_k
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        results = compression_retriever.invoke(query)
        outputs = []
        for result in results:
            output_data = {
            'page_content': result.page_content,
            'source': result.metadata.get('source'),
            'sectionTitle': result.metadata.get('sectionTitle')
            }
            
            outputs.append(output_data)
        return outputs
    
    def retrieveContentOnly(self, query: str):
        base_retriever = self.vector_db.as_retriever(
            search_kwargs={"k": self.retriever_top_k}
        )
        compressor = CrossEncoderReranker(
            model=self.rerank_model, top_n=self.reranker_top_k
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        results = compression_retriever.invoke(query)
        outputs = []
        for result in results:
            output_data = result.page_content
            
            outputs.append(output_data)
        return outputs