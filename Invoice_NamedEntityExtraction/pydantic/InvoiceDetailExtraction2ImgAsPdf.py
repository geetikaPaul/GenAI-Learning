from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
import logfire
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from typing import List

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

agent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp", api_key=os.getenv("Gemini_API_Key")
    ),
    system_prompt="Extract customer name, address, phone and items list with quantity",
)

# Does not work - image to pdf loses the meta tags of pdf hence, cannot be translated efficiently
img_file_path=os.path.expanduser("~/genAI/Invoice_NamedEntityExtraction/data/InvoiceSampleImg.pdf")


def get_text_from_pdf(path):
    pdf_loader = PyPDFLoader(path)
    docs = pdf_loader.load()
    return docs[0].page_content

invoice_data = get_text_from_pdf(os.path.expanduser(img_file_path))

with logfire.span("Calling model") as span:
  result = agent.run_sync(invoice_data)
  print(result.data)
    