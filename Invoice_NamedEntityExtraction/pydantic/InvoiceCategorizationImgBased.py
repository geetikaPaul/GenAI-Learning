from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from pydantic_ai import Agent
import logfire
from pydantic_ai.models.mistral import MistralModel
from typing import Optional
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Utils.ImagePrompt import ImageLoaderBase64

load_dotenv(override=True)
logfire.configure(token=os.getenv("Logfire_Write_Token"))

class Expenses(BaseModel):
    StoreName: Optional[str] = Field(description= "Pick the name on the top of the bill")
    Category: str = Field(description="if items are grocery items then grocery, electronics as electronic and so on")
    Amount: float

agent = Agent(
    model=MistralModel(
        model_name="pixtral-12b-2409", api_key=os.environ["MISTRAL_API_KEY"]
        ),
    system_prompt="Extract Total bill amount and category of products or store name",
    result_type= Expenses
)

image_prompt = ImageLoaderBase64(user_prompt="Extract total bill amount and category of bill", image_file_path=os.path.expanduser
                                 ("~/genAI/Invoice_NamedEntityExtraction/data/edekaBill.jpeg"))

result = agent.run_sync(image_prompt.encoded_message_with_image) 
invoice = result.data
print(invoice)