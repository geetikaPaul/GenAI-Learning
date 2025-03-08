from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logfire
from typing import List
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Utils.AgentBuilder import get_agent, get_prompt

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

class Product(BaseModel):
    Product_Name : str
    Product_Descritpion: str
    Quantity: float
    Price: float

class BillTo(BaseModel):
    Customer_Address: str
    Customer_Name : str
    Customer_Phone: str

class CompanyDetails(BaseModel):
    Company_Name : str
    Company_Address : str
    Company_phone: str

class Info(BaseModel):
    BillTo: BillTo
    Company: CompanyDetails
    Products : List[Product]
    Tax: float
    Total_cost: float

file_path =os.path.expanduser("~/genAI/Invoice/data/Betterinvoice.pdf")
agent = get_agent(file_path, "convert and replace fields with value NA or empty as 'hahaaa.", Info)

# logfire.instrument_openai()

# agent = Agent(
#     model=OpenAIModel(
#         model_name="llama3.2", base_url="http://localhost:11434"
#     ),
#     system_prompt="You are a helpful assistant."
# )
invoice_data = get_prompt(file_path=os.path.expanduser(file_path))

with logfire.span("Calling model") as span:
    # response = agent.run_sync("Tell me about India.")
    # print(response.data)

    result = agent.run_sync(invoice_data)
    print(result.data)

# response = agent.run_sync("Tell me about India.")
# print(response.data)
